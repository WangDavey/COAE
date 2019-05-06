# COAE
Clustering with Orthogonal AutoEncoder (COAE)
  * Wei WANG, Dan YANG, Feiyu CHEN, Yunsheng PANG, Sheng HUANG, Yongxin GE

We propose a novel dimensional reduction model, called Orthogonal AutoEncoder (OAE), which encourages orthogonality between the learned  embedding. Furthermore, we propose a joint deep Clustering framework based on Orthogonal AutoEncoder (COAE), this new framework is capable of extracting the latent embedding and predicting the clustering assignment simultaneously. COAE stacks a fully-connected clustering layer on top of OAE,  where the activation function of clustering layer is the multinomial logistic regression function. 

Tensorflow-based implementation of COAE

Requirements
-
To run COAE, you'll need Python 3.x and the following python packages:
 * tensorflow
 * keras
 * numpy
 * scikit-learn
 * munkres

Usage
-
All the datasets are uploaded in the Data. Specially, ROAE is for pre-train. It indicates that run K-means algorithm in the latent featurespace obtained from the proposed Orthogonal AutoEn-coder. In this project, you can directly run COAE without pre-train stage because well-trained parameters are provided.

```python3 ***.py```

Contact
-
wangwei0108@cqu.edu.cn
