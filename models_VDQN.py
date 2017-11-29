import tensorflow as tf
import edward as ed
from edward.models import Normal, Uniform
import numpy as np

####
# Q network
####


class VariationalQNetwork(object):

    def __init__(self, obssize, actsize, hiddendict, sess, tau=1.0, sigma=0.1,
                 scope='principle', Wpriorsigma=None, bpriorsigma=None,
                 optimizer=tf.train.AdamOptimizer(1e-3), sigma_rho=None):
        '''
        obssize: obssize of the env
        actsize: actsize of the env
        hiddendict: hidden layers of MLP e.g. [10, 5]
        sess: tf session to run the network
        tau: update parameter for target network,  tau = 1.0 for hard copy
        sigma: Q(s, a) = Q_theta(s, a) + N(0, 1) * sigma,
        generative model std for Q value
        scope: scope of the network
        '''
        self.obssize = obssize
        self.actsize = actsize
        self.hiddendict = hiddendict
        self.sess = sess
        self.tau = tau
        self.sigma = sigma
        self.optimizer = optimizer
        self.sigma_rho = sigma_rho

        with tf.variable_scope(scope):
            self._build_prior(self.obssize, self.hiddendict, self.actsize,
                              Wpriorsigma=Wpriorsigma, bpriorsigma=bpriorsigma)

            self._build_model_function(self.W, self.b, self.obssize,
                                       self.actsize, sigma=self.sigma)

            self._build_posterior(self.obssize, self.hiddendict, self.actsize,
                                  sigma_rho=self.sigma_rho)

            self._build_forward_computation(self.obssize, self.hiddendict,
                                            self.actsize)

            self._build_inference(optimizer=optimizer)

            self._build_assign_variables()

    def _build_prior(self, inputsize, hiddendict, outputsize, scope='prior',
                     Wpriorsigma=None, bpriorsigma=None):
        W, b = {}, {}
        Wshape, bshape = [], []
        with tf.variable_scope(scope):
            layerdict = [inputsize] + hiddendict + [outputsize]
            i = 0
            if Wpriorsigma is None:
                Wpriorsigma = (len(layerdict) - 1) * [0.1]
            if bpriorsigma is None:
                bpriorsigma = (len(layerdict) - 1) * [0.1]
            for h1, h2 in zip(layerdict[:-1], layerdict[1:]):
                W[i] = Uniform(low=tf.ones([h1, h2]) * (-10000),
                               high=tf.ones([h1, h2]) * 10000)
                b[i] = Uniform(low=tf.ones(h2) * (-10000),
                               high=tf.ones(h2) * 10000)
                Wshape.append([h1, h2])
                bshape.append([h2])
                i += 1

        self.W = W
        self.b = b
        self.Wshape = Wshape
        self.bshape = bshape

    def _build_model_function(self, W, b, inputsize, outputsize, scope='prior',
                              reuse=True, sigma=1.0):
        n = len(W.keys())
        with tf.variable_scope(scope, reuse=reuse):
            Xobs = tf.placeholder(tf.float32, [None, inputsize], name='Xobs')
            Xact = tf.placeholder(tf.int32, [None], name='Xact')
            Xact_onehot = tf.one_hot(Xact, outputsize, dtype=tf.float32)
            h = tf.nn.relu(tf.matmul(Xobs, W[0]) + b[0])
            for i in range(1, n - 1):
                h = tf.nn.relu(tf.matmul(h, W[i]) + b[i])
            h = tf.matmul(h, W[n - 1]) + b[n - 1]
            hchosen = tf.reduce_sum(tf.multiply(h, Xact_onehot), axis=1)
            Yact = Normal(loc=hchosen, scale=sigma, name='Y')
        self.Xobs = Xobs
        self.Xact = Xact
        self.Yact = Yact

    def _build_posterior(self, inputsize, hiddendict, outputsize,
                         scope='posterior', sigma_rho=None):
        qW, qb = {}, {}
        qWmu, qbmu = {}, {}
        qWrho, qbrho = {}, {}
        with tf.variable_scope(scope):
            layerdict = [inputsize] + hiddendict + [outputsize]
            i = 0
            for h1, h2 in zip(layerdict[:-1], layerdict[1:]):
                with tf.variable_scope('qW{0}'.format(i)):
                    if sigma_rho is None:
                        sigma_rho = np.log(np.exp(0.017) - 1.0)
                    qWmu[i] = tf.Variable(tf.random_uniform([h1, h2],
                                          - np.sqrt(3 / h1), np.sqrt(3 / h1)),
                                          name='loc')
                    qWrho[i] = tf.Variable(tf.random_uniform([h1, h2],
                                           sigma_rho, sigma_rho),
                                           name='scale', trainable=True)
                    qW[i] = Normal(loc=qWmu[i], scale=tf.nn.softplus(qWrho[i]))
                with tf.variable_scope('qb{0}'.format(i)):
                    qbmu[i] = tf.Variable(tf.random_uniform([h2], 0, 0),
                                          name='loc')
                    qbrho[i] = tf.Variable(tf.random_uniform([h2], sigma_rho,
                                           sigma_rho),
                                           name='scale', trainable=True)
                    qb[i] = Normal(loc=qbmu[i], scale=tf.nn.softplus(qbrho[i]))
                i += 1
        self.qW = qW
        self.qb = qb
        # following parameters are also used for
        # forward computation of Q values in rollout
        self.qWmu = qWmu
        self.qbmu = qbmu
        self.qWrho = qWrho
        self.qbrho = qbrho

    def _build_forward_computation(self, inputsize, hiddendict, outputsize):
        observation = tf.placeholder(tf.float32, [None, inputsize])
        Wnoise = {}
        bnoise = {}
        for i, W in self.qW.items():
            Wnoise[i] = tf.placeholder(tf.float32, [None] + list(W.shape))
        for i, b in self.qb.items():
            bnoise[i] = tf.placeholder(tf.float32, [None] + list(b.shape))

        # compute sampled parameters
        W_theta = {}
        b_theta = {}
        for i in self.qW.keys():
            W_theta[i] = self.qWmu[i] + tf.nn.softplus(
                self.qWrho[i]) * Wnoise[i]
        for i in self.qb.keys():
            b_theta[i] = self.qbmu[i] + tf.nn.softplus(
                self.qbrho[i]) * bnoise[i]

        # compute Q values
        n = len(W_theta.keys())

        h = tf.nn.relu(tf.einsum('ij,ijk->ik',
                       observation, W_theta[0]) + b_theta[0])
        for i in range(1, n - 1):
            h = tf.nn.relu(tf.einsum('ij,ijk->ik', h,
                           W_theta[i]) + b_theta[i])
        Qmu = tf.einsum('ij,ijk->ik', h,
                        W_theta[n - 1]) + b_theta[n - 1]

        self.W_theta = W_theta
        self.b_theta = b_theta
        self.observation = observation
        self.Wnoise = Wnoise
        self.bnoise = bnoise
        self.Qmu = Qmu

    def compute_Qvalue(self, observation, Wnoise, bnoise):
        feed_dict = {}
        for i in range(len(self.Wnoise.keys())):
            feed_dict[self.Wnoise[i]] = Wnoise[i]
            feed_dict[self.bnoise[i]] = bnoise[i]
        feed_dict[self.observation] = observation
        return self.sess.run(self.Qmu, feed_dict=feed_dict)

    def _build_inference(self, optimizer=tf.train.AdamOptimizer(1e-2)):
        inputdict = {}
        for k in self.W.keys():
            inputdict[self.W[k]] = self.qW[k]
        for k in self.b.keys():
            inputdict[self.b[k]] = self.qb[k]
        self.Yactph = tf.placeholder(tf.float32, [None])
        # self.inference = ed.KLqp(inputdict)
        self.inference = ed.KLqp(inputdict, data={self.Yact: self.Yactph})
        numiterations = 2000
        self.inference.initialize(n_iter=numiterations,
                                  scale={self.Yact: 1},
                                  optimizer=optimizer)

    def train_on_sample(self, observation, actions, targets):
        '''
        take gradient step only once
        '''
        return self.inference.update({self.Xobs: observation,
                                     self.Xact: actions, self.Yactph: targets})

    def fit_on_sample(self, observation, actions, targets):
        '''
        fit on sample data for multiple iterations until convergence
        '''
        lossrecord = []
        for ite in range(self.inference.n_iter):
            loss = self.inference.update({self.Xobs: observation,
                                         self.Xact: actions,
                                         self.Yactph: targets})
            print('ite %d loss %f' % (ite, loss['loss']))
            lossrecord.append(loss['loss'])
        return lossrecord

    def _build_assign_variables(self):
        # assign variables
        Wmuvalues = {k: tf.placeholder(tf.float32, v.shape)
                     for k, v in self.qWmu.items()}
        Wrhovalues = {k: tf.placeholder(tf.float32, v.shape)
                      for k, v in self.qWrho.items()}
        bmuvalues = {k: tf.placeholder(tf.float32, v.shape)
                     for k, v in self.qbmu.items()}
        brhovalues = {k: tf.placeholder(tf.float32, v.shape)
                      for k, v in self.qbrho.items()}
        assign_ops = []
        tau = self.tau
        for i in range(len(Wmuvalues.keys())):
            assign_ops.append(self.qWmu[i].assign(
                              tau * Wmuvalues[i] + (1 - tau) * self.qWmu[i]))
            assign_ops.append(self.qWrho[i].assign(
                              tau * Wrhovalues[i] + (1 - tau) * self.qWrho[i]))
            assign_ops.append(self.qbmu[i].assign(
                              tau * bmuvalues[i] + (1 - tau) * self.qbmu[i]))
            assign_ops.append(self.qbrho[i].assign(
                              tau * brhovalues[i] + (1 - tau) * self.qbrho[i]))
        self.assign_ops = assign_ops
        self.Wmuvalues = Wmuvalues
        self.Wrhovalues = Wrhovalues
        self.bmuvalues = bmuvalues
        self.brhovalues = brhovalues

    def assign_variables(self, Wmu, Wrho, bmu, brho):
        feed_dict = {}
        for i in range(len(self.qW.keys())):
            feed_dict[self.Wmuvalues[i]] = Wmu[i]
            feed_dict[self.Wrhovalues[i]] = Wrho[i]
            feed_dict[self.bmuvalues[i]] = bmu[i]
            feed_dict[self.brhovalues[i]] = brho[i]
        self.sess.run(self.assign_ops, feed_dict=feed_dict)

    def get_variables(self):
        Wmu, Wrho, bmu, brho = {}, {}, {}, {}
        for i in range(len(self.qWmu.keys())):
            Wmu[i] = self.sess.run(self.qWmu[i])
            Wrho[i] = self.sess.run(self.qWrho[i])
            bmu[i] = self.sess.run(self.qbmu[i])
            brho[i] = self.sess.run(self.qbrho[i])
        return Wmu, Wrho, bmu, brho


def update_target(Qtarget, Qaction):
    params = Qaction.get_variables()
    Qtarget.assign_variables(*params)


class NoiseSampler(object):

    def __init__(self, Wshape, bshape):
        assert len(Wshape) == len(bshape)
        self.Wshape = Wshape
        self.bshape = bshape

    def sample(self, numsamples):
        '''
        sample N(0, 1) noise using W, b shape of the network
        numsamples: num of samples to generate
        Wshape: list of shape of W,  [[10, 8], [8, 5]]
        bshape: list of shape of b,  [[8], [5]]
        '''
        Wshape = self.Wshape
        bshape = self.bshape
        Wnoise = {}
        bnoise = {}
        for i in range(len(Wshape)):
            Winput = [numsamples] + Wshape[i]
            Wnoise[i] = np.random.randn(*Winput)
            binput = [numsamples] + bshape[i]
            bnoise[i] = np.random.randn(*binput)
        return Wnoise, bnoise
