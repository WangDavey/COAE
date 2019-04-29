import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import argparse
from sklearn.cluster import KMeans
from munkres import Munkres
import scipy.io as scio
from scipy.linalg import orth
from keras.layers import UpSampling2D
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.data_utils import shuffle
from keras import backend as K
from datasets import load_mnist, load_usps
import scipy.io as sio


class DCN(object):
    def __init__(self, input_shape=[28, 28], kernel_size=[5, 5, 3], filters=[32, 64, 128], n_clusters=10,
                 batch_size=200, reg=None, dimension=10):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.filters = filters
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.reg = reg
        self.dimension = dimension
        weights = self._initialize_weights()
        self.saver = tf.train.Saver()

        # placeholder
        self.x = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.dropout_rate = tf.placeholder(tf.float32, [])

        # forward
        self.embedding = self.encoder(weights)
        self.decoder_net = self.decoder(self.embedding, weights)

        # loss function
        self.recon_loss = tf.reduce_sum(tf.pow(tf.subtract(self.decoder_net, self.x), 2.0))
        self.orth = tf.matmul(self.embedding, tf.transpose(self.embedding))
        self.orth_loss = 0.5 * (tf.reduce_sum(tf.pow(self.orth, 2)) - tf.reduce_sum(tf.pow(tf.diag_part(self.orth), 2)))
        
        self.loss = self.recon_loss / self.batch_size + 0.005 * self.orth_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1,
                self.filters[0]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_w1'] = tf.get_variable("enc_w1", shape=[self.kernel_size[1], self.kernel_size[1],
                self.filters[0], self.filters[1]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_w2'] = tf.get_variable("enc_w2", shape=[self.kernel_size[2], self.kernel_size[2],
                self.filters[1], self.filters[2]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)

        all_weights['emb_w0'] = tf.Variable(tf.truncated_normal(shape=[6272, self.dimension], stddev=0.04))
        all_weights['emb_w1'] = tf.Variable(tf.truncated_normal(shape=[self.dimension, 6272], stddev=0.04))

        all_weights['dec_w0'] = tf.get_variable("dec_w0", shape=[self.kernel_size[2], self.kernel_size[2],
                self.filters[1], self.filters[2]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_w1'] = tf.get_variable("dec_w1", shape=[self.kernel_size[1], self.kernel_size[1],
                self.filters[0], self.filters[1]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_w2'] = tf.get_variable("dec_w2", shape=[self.kernel_size[0], self.kernel_size[0], 1,
                self.filters[0]], initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        return all_weights

    def encoder(self, weights):
        layer1 = tf.nn.relu(tf.nn.conv2d(self.x, weights['enc_w0'], strides=[1, 1, 1, 1], padding='SAME'))
        layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer2 = tf.nn.relu(tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1, 1, 1, 1], padding='SAME'))
        layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer3 = tf.nn.relu(tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1, 1, 1, 1], padding='SAME'))
        #layer3 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        layer3 = tf.reshape(layer3, [-1, layer3.shape[1]*layer3.shape[2]*layer3.shape[3]])
        layer3 = tf.nn.dropout(layer3, self.dropout_rate)
        embedding = tf.matmul(layer3, weights['emb_w0'])
        return embedding

    def decoder(self, x, weights):
        layer1 = tf.nn.relu(tf.matmul(x, weights['emb_w1']))
        layer1 = tf.reshape(layer1, [self.batch_size, 7, 7, self.filters[2]])
        layer2 = tf.nn.relu(tf.nn.conv2d_transpose(layer1, weights['dec_w0'], output_shape=[self.batch_size, 7, 7, 64],
                                                   strides=[1, 1, 1, 1], padding='SAME'))
        layer2 = UpSampling2D((2, 2))(layer2)
        layer3 = tf.nn.relu(tf.nn.conv2d_transpose(layer2, weights['dec_w1'], output_shape=[self.batch_size, 14, 14, 32],
                                                   strides=[1, 1, 1, 1], padding='SAME'))
        layer3 = UpSampling2D((2, 2))(layer3)
        layer4 = tf.nn.conv2d_transpose(layer3, weights['dec_w2'], output_shape=[self.batch_size, 28, 28, 1],
                                        strides=[1, 1, 1, 1], padding='SAME')
        return layer4

    def finetune_fit(self, X, lr, dr):
        total_loss, kmeans_input, orth_loss, _ = self.sess.run((self.loss, self.embedding, self.orth_loss, self.optimizer), feed_dict={self.x: X,
                                      self.learning_rate: lr, self.dropout_rate: dr})
        return total_loss, kmeans_input

    def forward(self, X, lr, dr):
        features = self.sess.run(self.embedding, feed_dict={self.x: X, self.learning_rate: lr, self.dropout_rate: dr})
        #features = orth(features)
        return features

    def save_model(self):
        save_path = self.saver.save(self.sess, 'pretrained/dcn-mnist.ckpt')
        print("model saved in file: %s" % save_path)

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    dis = [gt_s[i] - c_x[i] for i in range(len(gt_s))]
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate, dis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCN')
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--ft_times', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--display_step', default=1, type=int)
    parser.add_argument('--dimension', default=10, type=int)
    parser.add_argument('--dropout_rate', default=0.9, type=float)
    args = parser.parse_args()

    # 数据处理
    Img_mnist, Label_mnist = load_mnist()
    print('Data preprocessing has done!')

    # 网络初始化
    DCN_kmeans = DCN(input_shape=[28, 28], kernel_size=[5, 5, 3], filters=[32, 64, 128], n_clusters=10,
                     batch_size=args.batch_size, reg=None, dimension=args.dimension)
    DCN_kmeans.initlization()
    print('DCN-Net initlization has done!')

    # 参数声明
    iter_ft = 0
    index_ft = 0

    Img_mnist, Label_mnist = shuffle(Img_mnist, Label_mnist)  # shuffle the data
    for iter_ft in range(args.ft_times):  # ft_time: 迭代次数
        if iter_ft > 5 and iter_ft % 20 == 0:
            args.learning_rate /= 5
        for index_ft in range(0, Img_mnist.shape[0], args.batch_size):  # 0-65000, step=200
            start = index_ft
            if index_ft + args.batch_size > Img_mnist.shape[0]:
                start = Img_mnist.shape[0] - args.batch_size
            Loss, kmeans_input = DCN_kmeans.finetune_fit(Img_mnist[start:start+args.batch_size], args.learning_rate, args.dropout_rate)
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(kmeans_input)
            y_labels = kmeans.labels_      

            if index_ft == 40960:  # 在每个epoch的前40960个数据测试
                features = DCN_kmeans.forward(Img_mnist[:41216], 0, 1)
                kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(features)
                y_pred = kmeans.labels_
                missrate_x, dis_x = err_rate(Label_mnist[:41216], y_pred)
                acc = 1 - missrate_x
                if iter_ft == 0:
                    save_acc = np.array(acc)
                else:
                    save_acc = np.append(save_acc, acc)
                dis_x = np.array(dis_x)
                print("epoch: %d" % iter_ft, "cost: %.8f" % Loss, "acc: %.4f" % acc)

        
