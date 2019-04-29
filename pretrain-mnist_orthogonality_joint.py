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

from numpy.random import seed
seed(110)
from tensorflow import set_random_seed
set_random_seed(5)
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
        self.iter = 0
        weights = self._initialize_weights()
        #self.saver = tf.train.Saver()
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("cluster_layer"))])

        # placeholder
        self.x = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.dropout_rate = tf.placeholder(tf.float32, [])

        # forward
        self.embedding = self.encoder(weights)
        self.p = self.clusteringlayer(weights)
        #print(self.clusterlayer.shape)
        #self.p = tf.argmax(self.clusterlayer, 1)
        #self.p = self.clusterlayer
        #print(self.p.shape)
        #self.target = self.p / tf.sqrt(tf.reshape(tf.reduce_sum(self.p, 0), [1, 10]))
        self.target = tf.pow(self.p, 1) / tf.pow(tf.reshape(tf.reduce_sum(self.p, 0), [1, 10]), 0.5)
        #print(tf.reshape(tf.reduce_sum(self.p, 0), [1,10]).shape)
        #print(self.target.shape)
        self.q = tf.transpose(tf.transpose(self.target) / tf.reshape(tf.reduce_sum(self.target, 1), [1, self.batch_size]))
        #print(self.q.shape)

        self.decoder_net = self.decoder(self.embedding, weights)

        # loss function
        self.kl = tf.reduce_sum(tf.reduce_sum(tf.multiply(self.q, tf.log(self.p)), 1), 0)
        #print(self.t.shape)
        self.recon_loss = tf.reduce_sum(tf.pow(tf.subtract(self.decoder_net, self.x), 2.0))
        #self.orth = tf.matmul(self.embedding, tf.transpose(self.embedding))
        self.orth = tf.matmul(tf.transpose(self.embedding), self.embedding)
        #self.orth = tf.matmul(self.embedding, tf.transpose(self.embedding))
        #self.orth_loss = tf.reduce_sum(tf.pow((self.orth - tf.eye(self.batch_size)), 2))
        self.orth_loss = tf.reduce_sum(tf.pow((self.orth - tf.eye(self.dimension)), 2))
        #self.orth_loss = 0.5 * (tf.reduce_sum(tf.pow(self.orth, 2)) - tf.reduce_sum(tf.pow(tf.diag_part(self.orth), 2)))
        #self.orth_loss = 0.5 * (tf.reduce_sum(tf.pow(self.orth, 2)) - tf.reduce_sum(tf.pow(tf.diag_part(self.orth), 2)))
        self.loss = self.recon_loss / self.batch_size + 0.005 * self.orth_loss - 5 * self.kl / self.batch_size
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('orthogonal_loss', self.orth_loss)
        tf.summary.scalar('crossentropy_loss', self.kl)
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter('./MNIST_Log', graph=tf.get_default_graph())

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
        #all_weights['clu_w0'] = tf.Variable(tf.truncated_normal(shape=[self.dimension, self.dimension], stddev=0.04))


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
    
    # def Orthonorm(self, batch_size):
    #     Orth_layer = tf.Variable(tf.ones([self.dimension, self.dimension],tf.float32), name = 'Orth_layer')
    #     Orth_embedding = tf.matmul(Orth_layer, self.embedding)
    #     return Orth_embedding, Orth_layer
    def clusteringlayer(self, weights):
        cluster_layer = tf.Variable(tf.truncated_normal(shape=[self.dimension, self.dimension], stddev=0.001, seed=5), name = 'cluster_layer')
        #soft_cluster = tf.nn.softmax(tf.matmul(self.embedding, weights['clu_w0']))
        soft_cluster = tf.nn.softmax(tf.matmul(self.embedding, cluster_layer))
        return soft_cluster

    def finetune_fit(self, X, lr, dr):
        total_loss, kmeans_input, orth_loss, q, _ = self.sess.run((self.loss, self.embedding, self.orth_loss, self.q, self.optimizer), feed_dict={self.x: X, self.learning_rate: lr, self.dropout_rate: dr})
        #print(orth_loss)
        #kmeans_input = orth(kmeans_input)
        #print(q.shape)
        #print('################')
        #q = q.argmax(1)
        #print(q)
        return total_loss, kmeans_input

    def forward(self, X, lr, dr):
        features, q, reconstruction, summary = self.sess.run((self.embedding, self.q, self.decoder_net, self.merged_summary_op), feed_dict={self.x: X, self.learning_rate: lr, self.dropout_rate: dr})
        #features = orth(features)
        #print(q.shape)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        q = q.argmax(1)
        #print(q.shape)
        return features, q, reconstruction
    

    def save_model(self):
        save_path = self.saver.save(self.sess, 'pretrained/dcn-mnist.ckpt')
        print("model saved in file: %s" % save_path)

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)
    
    def restore(self):
        self.saver.restore(self.sess, 'pretrained/dcn-mnist.ckpt')
        print("model restored")


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
    # print(dis[:100])
    # print(len(dis))

    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate, dis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCN')
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--ft_times', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--display_step', default=1, type=int)
    parser.add_argument('--dimension', default=10, type=int)
    parser.add_argument('--dropout_rate', default=0.9, type=float)
    args = parser.parse_args()

    # 数据处理
    # mnist = input_data.read_data_sets('./Data/', one_hot=False)
    # Img_train = mnist.train.images
    # Img_test = mnist.test.images
    # Img = np.concatenate((Img_train, Img_test)).astype(float)  # Img.shape=(65000, 784)
    # Img = np.reshape(Img, [-1, 28, 28, 1])   # Img.shape=(65000, 28, 28, 1)
    # Label_train = mnist.train.labels
    # Label_test = mnist.test.labels
    # Label = np.concatenate((Label_train, Label_test))  # Label.shape=(65000,)
    # # Label = np.squeeze(Label)
    Img_mnist, Label_mnist = load_mnist()
    print('Data preprocessing has done!')

    # 网络初始化
    DCN_kmeans = DCN(input_shape=[28, 28], kernel_size=[5, 5, 3], filters=[32, 64, 128], n_clusters=10,
                     batch_size=args.batch_size, reg=None, dimension=args.dimension)
    DCN_kmeans.initlization()
    DCN_kmeans.restore()
    print('DCN-Net initlization has done!')

    # 参数声明
    iter_ft = 0
    index_ft = 0
    test_ft = 0
    s_ = np.zeros([65000, 10])  # s_.shape=(65000, 10)

    Img_mnist, Label_mnist = shuffle(Img_mnist, Label_mnist)  # shuffle the data
    print(Img_mnist.shape)
    #Img_mnist, Label_mnist = Img, Label
    for iter_ft in range(args.ft_times):  # ft_time: 迭代次数
        total_loss = 0
        if iter_ft > 5 and iter_ft % 20 == 0:
            args.learning_rate /= 5
        for index_ft in range(0, Img_mnist.shape[0], args.batch_size):  # 0-65000, step=200
            start = index_ft
            if index_ft + args.batch_size > Img_mnist.shape[0]:
                start = Img_mnist.shape[0] - args.batch_size
            # kmeans_input.shape=(200, 10)
            Loss, kmeans_input = DCN_kmeans.finetune_fit(Img_mnist[start:start+args.batch_size], args.learning_rate, args.dropout_rate)
            # kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(kmeans_input)
            # y_labels = kmeans.labels_  # y_labels.shape=(batch_size,)
            # M = kmeans.cluster_centers_
            # save model and m
            # if start == Img_mnist.shape[0] - args.batch_size:
            #     M = np.transpose(M)
            #     scio.savemat('./M/M.mat', {'ui': M})
            #     #DCN_kmeans.save_model()
            

            if index_ft == 69888:  # 在每个epoch的前30000个数据测试
                #features = DCN_kmeans.forward(Img_mnist[:10496], args.learning_rate, args.dropout_rate)
                for test_ft in range(0, 69888, args.batch_size):
                    test_start = test_ft
                    if test_ft + args.batch_size > index_ft:
                        test_start = index_ft - args.batch_size
                    features, q, reconstruction = DCN_kmeans.forward(Img_mnist[test_start:test_start + args.batch_size], 0, 1)
                    #print(features.shape)
                    y_pred_batch = q
                    features_ = features#
                    reconstruction_ = reconstruction
                    if test_ft == 0:
                        y_pred = y_pred_batch
                        x_pred = features_#
                        r_pred = reconstruction_
                    else:
                        y_pred = np.hstack((y_pred, y_pred_batch))
                        x_pred = np.vstack((x_pred, features_))#
                        r_pred = np.vstack((r_pred, reconstruction_))
                #kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(features)
                #missrate_x = err_rate(Label_mnist[:25856], kmeans.labels_)
                #print(x_pred.shape)
                #print(y_pred.shape)
                #missrate_x = err_rate(Label_mnist[:index_ft], y_pred)
                missrate_x, dis_x = err_rate(Label_mnist[:69888], y_pred)
                acc = 1 - missrate_x
                if iter_ft == 0:
                    save_acc = np.array(acc)
                else:
                    save_acc = np.append(save_acc, acc)
                #print(save_acc)
                dis_x = np.array(dis_x)
                print("epoch: %d" % iter_ft, "cost: %.8f" % Loss, "acc: %.4f" % acc)
        sio.savemat('./I/i.mat', {'ii': Img_mnist[:69888]})
        sio.savemat('./S/s.mat', {'si': x_pred})
        sio.savemat('./M/M.mat', {'ui': Label_mnist[:69888]})
        sio.savemat('./D/D.mat', {'di': dis_x})
        sio.savemat('./A/A.mat', {'ai': save_acc})
        sio.savemat('./Y/Y.mat', {'yi': y_pred})
        sio.savemat('./R/R.mat', {'ri': r_pred})

                # with open('./mnist_results4.txt', 'a') as f:
                #     f.write(str(iter_ft) + ' ' + str(Loss) + ' ' + str(acc)+'\n')
        # for index_ft in range(0, Img_mnist.shape[0], args.batch_size):
        #     start = index_ft
        #     if index_ft + args.batch_size > Img_mnist.shape[0]:
        #         start = Img_mnist.shape[0] - args.batch_size
        #     features = DCN_kmeans.forward(Img_mnist[start:start + args.batch_size], 0, 1)
        #     kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(features)
        #     y_pred_batch = kmeans.labels_
        #     if index_ft == 0:
        #         y_pred = y_pred_batch
        #     else:
        #         y_pred = np.hstack((y_pred, y_pred_batch))




        # missrate_x = err_rate(Label_mnist, y_pred)
        # acc = 1 - missrate_x
        # print("epoch: %.1d" % iter_ft, "acc: %.4f" % acc)

    #         if iter_ft == args.ft_times - 1:
    #             if index_ft + args.batch_size > Img_mnist.shape[0]:
    #                 for j in range(args.batch_size):
    #                     for i in range(10):
    #                         if y_labels[j] == i:
    #                             s_[Img_mnist.shape[0] - args.batch_size + j][i] = 1
    #             else:
    #                 for j in range(index_ft, index_ft+args.batch_size):
    #                     for i in range(10):
    #                         if y_labels[j-index_ft] == i:
    #                             s_[j][i] = 1
    # s_ = np.transpose(s_)
    # scio.savemat('./S/s.mat', {'si': s_})
