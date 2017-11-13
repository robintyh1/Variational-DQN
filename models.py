import tensorflow as tf
import edward as ed
from edward.models import Normal,PointMass,Uniform
import numpy as np

####
# Q network
####

class VariationalQNetwork(object):

	def __init__(self,obssize,actsize,hiddendict,sess,tau=1.0,sigma=0.1,scope='principle',
				Wpriorsigma=None,bpriorsigma=None,optimizer=tf.train.AdamOptimizer(1e-3),sigma_rho=None):
		'''
		obssize: obssize of the env
		actsize: actsize of the env
		hiddendict: hidden layers of MLP e.g. [10,5]
		sess: tf session to run the network
		tau: update parameter for target network, tau = 1.0 for hard copy
		sigma: Q(s,a) = Q_theta(s,a) + N(0,1) * sigma, generative model std for Q value
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
			self._build_prior(self.obssize,self.hiddendict,self.actsize,Wpriorsigma=Wpriorsigma,bpriorsigma=bpriorsigma)

			self._build_model_function(self.W,self.b,self.obssize,self.actsize,sigma=self.sigma)

			self._build_posterior(self.obssize,self.hiddendict,self.actsize,sigma_rho=self.sigma_rho)

			self._build_forward_computation(self.obssize,self.hiddendict,self.actsize)

			self._build_inference(optimizer=optimizer)

			self._build_assign_variables()

	def _build_prior(self,inputsize,hiddendict,outputsize,scope='prior',Wpriorsigma=None,bpriorsigma=None):
		W,b = {},{}
		Wshape,bshape = [],[]
		with tf.variable_scope(scope):
			layerdict = [inputsize] + hiddendict + [outputsize]
			i = 0
			if Wpriorsigma is None:
				Wpriorsigma = (len(layerdict)-1) * [0.1]
			if bpriorsigma is None:
				bpriorsigma = (len(layerdict)-1) * [0.1]
			for h1,h2 in zip(layerdict[:-1],layerdict[1:]):
				#W[i] = Normal(loc=tf.zeros([h1,h2]),scale=Wpriorsigma[i]*tf.ones([h1,h2]))
				#b[i] = Normal(loc=tf.zeros(h2),scale=bpriorsigma[i]*tf.ones(h2))
				W[i] = Uniform(low=tf.ones([h1,h2])*(-100),high=tf.ones([h1,h2])*100)
				b[i] = Uniform(low=tf.ones(h2)*(-100),high=tf.ones(h2)*100)
				Wshape.append([h1,h2])
				bshape.append([h2])
				i += 1

		self.W = W
		self.b = b
		self.Wshape = Wshape
		self.bshape = bshape

	def _build_model_function(self,W,b,inputsize,outputsize,scope='prior',reuse=True,sigma=1.0):
		n = len(W.keys())
		with tf.variable_scope(scope,reuse=reuse):
			Xobs = tf.placeholder(tf.float32,[None,inputsize],name='Xobs')
			Xact = tf.placeholder(tf.int32,[None],name='Xact')
			Xact_onehot = tf.one_hot(Xact,outputsize,dtype=tf.float32) 		
			h = tf.nn.relu(tf.matmul(Xobs,W[0]) + b[0])
			for i in range(1,n-1):
				h = tf.nn.relu(tf.matmul(h,W[i]) + b[i])
			h = tf.matmul(h,W[n-1]) + b[n-1]
			hchosen = tf.reduce_sum(tf.multiply(h,Xact_onehot),axis=1)
			Yact =  Normal(loc=hchosen,scale=sigma,name='Y')
		
		self.Xobs = Xobs
		self.Xact = Xact
		self.Yact = Yact

	def _build_posterior(self,inputsize,hiddendict,outputsize,scope='posterior',sigma_rho=None):
		qW,qb = {},{}
		qWmu,qbmu = {},{}
		qWrho,qbrho = {},{}
		with tf.variable_scope(scope):
			layerdict = [inputsize] + hiddendict + [outputsize]
			i = 0
			for h1,h2 in zip(layerdict[:-1],layerdict[1:]):
				with tf.variable_scope('qW{0}'.format(i)):
					if sigma_rho is None:
						sigma_rho = np.log(np.exp(0.017)-1.0)
					#sigma_rho = -10
					qWmu[i] = tf.Variable(tf.random_uniform([h1,h2],-np.sqrt(3/h1),np.sqrt(3/h1)),name='loc')
					qWrho[i] = tf.Variable(tf.random_uniform([h1,h2],sigma_rho,sigma_rho),name='scale',trainable=True)
					#qWmu[i] = tf.Variable(tf.random_normal([h1,h2],stddev=0.5/np.sqrt(h1+h2)),name='loc')
					#qWrho[i] = tf.Variable(tf.random_normal([h1,h2],stddev=0.5/np.sqrt(h1)),name='scale')
					#qWrho[i] = tf.Variable(tf.random_normal([h1,h2],mean=np.log(np.exp(0.5/np.sqrt(h1+h2))-1),stddev=0.0001),name='scale')
					qW[i] = PointMass(params=qWmu[i])
				with tf.variable_scope('qb{0}'.format(i)):
					qbmu[i] = tf.Variable(tf.random_uniform([h2],0,0),name='loc')
					qbrho[i] = tf.Variable(tf.random_uniform([h2],sigma_rho,sigma_rho),name='scale',trainable=True)
					#qbmu[i] = tf.Variable(tf.random_normal([h2],stddev=0.5/np.sqrt(h1+h2)),name='loc')
					#qbrho[i] = tf.Variable(tf.random_normal([h2]),name='scale')
					#qbrho[i] = tf.Variable(tf.random_normal([h2],mean=np.log(np.exp(0.5/np.sqrt(h1+h2))-1),stddev=0.0001),name='scale')
					qb[i] = PointMass(params=qbmu[i])
				i += 1
		
		self.qW = qW
		self.qb = qb
		# following parameters are also used for 
		# forward computation of Q values in rollout
		self.qWmu = qWmu
		self.qbmu = qbmu
		self.qWrho = qWrho
		self.qbrho = qbrho

	def _build_forward_computation(self,inputsize,hiddendict,outputsize):
		observation = tf.placeholder(tf.float32,[None,inputsize])
		Wnoise = {}
		bnoise = {}
		for i,W in self.qW.items():
			Wnoise[i] = tf.placeholder(tf.float32,[None]+list(W.shape))
		for i,b in self.qb.items():
			bnoise[i] = tf.placeholder(tf.float32,[None]+list(b.shape))

		# compute sampled parameters		
		W_theta = {}
		b_theta = {}
		for i in self.qW.keys():
			W_theta[i] = self.qWmu[i] + tf.nn.softplus(self.qWrho[i]) * Wnoise[i] # 0
		for i in self.qb.keys():
			b_theta[i] = self.qbmu[i] + tf.nn.softplus(self.qbrho[i]) * bnoise[i] # 0

		# compute Q values
		n = len(W_theta.keys())

		h = tf.nn.relu(tf.einsum('ij,ijk->ik',observation,W_theta[0]) + b_theta[0])
		for i in range(1,n-1):
			h = tf.nn.relu(tf.einsum('ij,ijk->ik',h,W_theta[i]) + b_theta[i])
		Qmu = tf.einsum('ij,ijk->ik',h,W_theta[n-1]) + b_theta[n-1]

		self.W_theta = W_theta
		self.b_theta = b_theta
		self.observation = observation
		self.Wnoise = Wnoise
		self.bnoise = bnoise
		self.Qmu = Qmu

	'''
	def _build_forward_computation(self,inputsize,hiddendict,outputsize):
		observation = tf.placeholder(tf.float32,[None,inputsize])
		Wnoise = {}
		bnoise = {}
		for i,W in self.qW.items():
			Wnoise[i] = tf.placeholder(tf.float32,[None]+list(W.shape))
		for i,b in self.qb.items():
			bnoise[i] = tf.placeholder(tf.float32,[None]+list(b.shape))

		# compute sampled parameters
		W_theta = {}
		b_theta = {}
		for i in self.qW.keys():
			W_theta[i] = self.qWmu[i] + tf.nn.softplus(self.qWrho[i]) * Wnoise[i]
		for i in self.qb.keys():
			b_theta[i] = self.qbmu[i] + tf.nn.softplus(self.qbrho[i]) * bnoise[i]

		# compute Q values
		#print(W_theta[0].shape,observation.shape,b_theta[0].shape)
		#h = tf.squeeze(tf.tensordot(observation,W_theta[0],[[1],[1]]),1)
		#print(h.shape)

		n = len(W_theta.keys())

		h = tf.nn.relu(tf.tensordot(observation,W_theta[0],[[0,1],[0,1]]) + b_theta[0])
		for i in range(1,n-1):
			h = tf.nn.relu(tf.tensordot(h,W_theta[i],[[0,1],[0,1]]) + b_theta[i])
		Qmu = tf.tensordot(h,W_theta[n-1],[[0,1],[0,1]]) + b_theta[n-1]	

		self.observation = observation
		self.Wnoise = Wnoise
		self.bnoise = bnoise
		self.Qmu = Qmu
	'''

	def compute_Qvalue(self,observation,Wnoise,bnoise):
		feed_dict = {}
		for i in range(len(self.Wnoise.keys())):
			feed_dict[self.Wnoise[i]] = Wnoise[i]
			feed_dict[self.bnoise[i]] = bnoise[i]
		feed_dict[self.observation] = observation
		return self.sess.run(self.Qmu,feed_dict=feed_dict)

	def _build_inference(self,optimizer=tf.train.AdamOptimizer(1e-2)):
		inputdict = {}
		for k in self.W.keys():
			inputdict[self.W[k]] = self.qW[k]
		for k in self.b.keys():
			inputdict[self.b[k]] = self.qb[k]
		self.Yactph = tf.placeholder(tf.float32,[None])
		#self.inference = ed.KLqp(inputdict)
		self.inference = ed.MAP(inputdict,data={self.Yact:self.Yactph})
		numiterations = 2000
		self.inference.initialize(n_iter=numiterations,scale={self.Yact:1},optimizer=optimizer)

	def train_on_sample(self,observation,actions,targets):
		'''
		take gradient step only once
		'''
		return self.inference.update({self.Xobs:observation,self.Xact:actions,self.Yactph:targets})

	def fit_on_sample(self,observation,actions,targets):
		'''
		fit on sample data for multiple iterations until convergence
		'''
		lossrecord = []
		for ite in range(self.inference.n_iter):
			loss = self.inference.update({self.Xobs:observation,self.Xact:actions,self.Yactph:targets})
			print('ite %d loss %f' %(ite,loss['loss']))
			lossrecord.append(loss['loss'])
		return lossrecord

	def _build_assign_variables(self):
		# assign variables
		Wmuvalues = {k:tf.placeholder(tf.float32,v.shape) for k,v in self.qWmu.items()}
		Wrhovalues = {k:tf.placeholder(tf.float32,v.shape) for k,v in self.qWrho.items()}
		bmuvalues = {k:tf.placeholder(tf.float32,v.shape) for k,v in self.qbmu.items()}
		brhovalues = {k:tf.placeholder(tf.float32,v.shape) for k,v in self.qbrho.items()}
		assign_ops = []
		tau = self.tau
		for i in range(len(Wmuvalues.keys())):
			assign_ops.append(self.qWmu[i].assign(tau*Wmuvalues[i]+(1-tau)*self.qWmu[i]))
			assign_ops.append(self.qWrho[i].assign(tau*Wrhovalues[i]+(1-tau)*self.qWrho[i]))
			assign_ops.append(self.qbmu[i].assign(tau*bmuvalues[i]+(1-tau)*self.qbmu[i]))
			assign_ops.append(self.qbrho[i].assign(tau*brhovalues[i]+(1-tau)*self.qbrho[i]))
		
		self.assign_ops = assign_ops
		self.Wmuvalues = Wmuvalues
		self.Wrhovalues = Wrhovalues
		self.bmuvalues = bmuvalues
		self.brhovalues = brhovalues

	def assign_variables(self,Wmu,Wrho,bmu,brho):
		feed_dict = {}
		for i in range(len(self.qW.keys())):
			feed_dict[self.Wmuvalues[i]] = Wmu[i]
			feed_dict[self.Wrhovalues[i]] = Wrho[i]
			feed_dict[self.bmuvalues[i]] = bmu[i]
			feed_dict[self.brhovalues[i]] = brho[i]
		self.sess.run(self.assign_ops,feed_dict=feed_dict)

	def get_variables(self):
		Wmu,Wrho,bmu,brho = {},{},{},{}
		for i in range(len(self.qWmu.keys())):
			Wmu[i] = self.sess.run(self.qWmu[i])
			Wrho[i] = self.sess.run(self.qWrho[i])
			bmu[i] = self.sess.run(self.qbmu[i])
			brho[i] = self.sess.run(self.qbrho[i])
		return Wmu,Wrho,bmu,brho


def update_target(Qtarget,Qaction):
	params = Qaction.get_variables()
	Qtarget.assign_variables(*params)


class NoiseSampler(object):

	def __init__(self,Wshape,bshape):
		assert len(Wshape) == len(bshape)
		self.Wshape = Wshape
		self.bshape = bshape

	def sample(self,numsamples):
		'''
		sample N(0,1) noise using W,b shape of the network
		numsamples: num of samples to generate
		Wshape: list of shape of W, [[10,8],[8,5]]
		bshape: list of shape of b, [[8],[5]]
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
		return Wnoise,bnoise

test=False
if test:
	with tf.Session() as sess:
		bnoiseprior = [1] * 3
		Wnoiseprior = [1] * 3
		Qnet = VariationalQNetwork(4,2,[20,20],sess,Wpriorsigma=Wnoiseprior,bpriorsigma=bnoiseprior,
				scope='principle',sigma=.1,sigma_rho=-10,optimizer=tf.train.AdamOptimizer(1e-3))
		#Qtarget = VariationalQNetwork(4,2,[20,20],sess,scope='target')
		sess.run(tf.global_variables_initializer())
		#N = 1000
		#observation = np.random.randn(N,10)
		noisesampler = NoiseSampler(Qnet.Wshape,Qnet.bshape)
		#Wnoise,bnoise = noisesampler.sample(N)
		#output = Qnet.compute_Qvalue(observation,Wnoise,bnoise)
		#print(output.shape)

		#for ite in range(30):
		#	actions = np.random.randint(low=0,high=8,size=(N))
		#	target = np.random.randn(N)
		#	loss = Qnet.train_on_sample(observation,actions,target)
		#	print(loss)

		#update_target(Qtarget,Qnet)

		'''
		see if sampled theta differs much
		'''
		Wnoise,bnoise = noisesampler.sample(1)
		feed_dict = {}
		for i in range(len(Qnet.Wnoise.keys())):
			feed_dict[Qnet.Wnoise[i]] = Wnoise[i]
			feed_dict[Qnet.bnoise[i]] = bnoise[i]
		W1 = Qnet.sess.run(Qnet.W_theta[0],feed_dict=feed_dict)
		b1 = Qnet.sess.run(Qnet.b_theta[0],feed_dict=feed_dict)

		Wnoise,bnoise = noisesampler.sample(1)
		feed_dict = {}
		for i in range(len(Qnet.Wnoise.keys())):
			feed_dict[Qnet.Wnoise[i]] = Wnoise[i]
			feed_dict[Qnet.bnoise[i]] = bnoise[i]
		W2 = Qnet.sess.run(Qnet.W_theta[0],feed_dict=feed_dict)
		b2 = Qnet.sess.run(Qnet.b_theta[0],feed_dict=feed_dict)

		print(np.sum((W1-W2)**2))
		print(np.sum((b1-b2)**2))

		'''
		see if large number of generated data can reduce the effect of prior
		and turn MAP to MLE
		also see if VI can approximate this point distributed MAP well
		'''
		import gym
		env = gym.make('CartPole-v0')
		episodes = 10000
		obs_record = []
		action_record = []
		reward_record = []
		done_record = []
		nextobs_record = []
		for e in range(episodes):
			obs = env.reset()
			done = False
			while not done:
				action = env.action_space.sample()
				nextobs,reward,done,_ = env.step(action)
				obs_record.append(obs)
				action_record.append(action)
				reward_record.append(reward)
				done_ = 1 if done else 0
				done_record.append(done_)
				nextobs_record.append(nextobs)

				obs = nextobs
				if done:
					break
			#print('episode %d' %(e))
		print('convert to arrays, data size %d' %(len(action_record)))
		obs_record = np.array(obs_record)
		action_record = np.array(action_record)
		reward_record = np.array(reward_record)
		done_record = np.array(done_record)
		nextobs_record = np.array(nextobs_record)

		batchsize = action_record.size
		print('sampling noise')
		Wnoise,bnoise = noisesampler.sample(batchsize)
		# compute target value
		gamma = .99
		print('computing target')
		Qall = Qnet.compute_Qvalue(nextobs_record,Wnoise,bnoise)
		Qtarget = gamma * np.max(Qall,axis=1) * (1-done_record) + reward_record
		# bellman error before fitting
		Qpred = Qnet.compute_Qvalue(obs_record,Wnoise,bnoise)
		Qpredact = Qpred[np.arange(batchsize),action_record]
		print('Bellman error before fitting %f' %(np.mean((Qpredact-Qtarget)**2)))
		# fit on data		
		print('fit on data')
		lossrecord = Qnet.fit_on_sample(obs_record,action_record,Qtarget)
		print('loss init %f, loss final %f' %(lossrecord[0],lossrecord[-1]))
		# bellman error after fitting
		Qpred = Qnet.compute_Qvalue(obs_record,Wnoise,bnoise)
		Qpredact = Qpred[np.arange(batchsize),action_record]
		print('Bellman error after fitting %f' %(np.mean((Qpredact-Qtarget)**2)))	

		Wnoise,bnoise = noisesampler.sample(1)
		feed_dict = {}
		for i in range(len(Qnet.Wnoise.keys())):
			feed_dict[Qnet.Wnoise[i]] = Wnoise[i]
			feed_dict[Qnet.bnoise[i]] = bnoise[i]
		W1 = Qnet.sess.run(Qnet.W_theta[0],feed_dict=feed_dict)
		b1 = Qnet.sess.run(Qnet.b_theta[0],feed_dict=feed_dict)

		Wnoise,bnoise = noisesampler.sample(1)
		feed_dict = {}
		for i in range(len(Qnet.Wnoise.keys())):
			feed_dict[Qnet.Wnoise[i]] = Wnoise[i]
			feed_dict[Qnet.bnoise[i]] = bnoise[i]
		W2 = Qnet.sess.run(Qnet.W_theta[0],feed_dict=feed_dict)
		b2 = Qnet.sess.run(Qnet.b_theta[0],feed_dict=feed_dict)

		print(np.sum((W1-W2)**2))
		print(np.sum((b1-b2)**2))


if False:
	with tf.Session() as sess:
		theta = Normal(loc=tf.zeros(10),scale=tf.ones(10))#Uniform(-100*tf.ones(10),100*tf.ones(10))
		x = tf.placeholder(tf.float32,[1000,10])
		var = tf.Variable(tf.random_normal([10]))
		qtheta = PointMass(params=var)
		y = Normal(ed.dot(x,theta),0.1*tf.ones(1000))
		yph = tf.placeholder(tf.float32,[1000])

		xdata = np.random.randn(1000,10)
		theta_true = np.random.randn(10)
		ydata = np.dot(xdata,theta_true) + np.random.randn(1000) * 0.1

		inference = ed.MAP({theta:qtheta},data={y:yph})
		inference.initialize(n_iter=100,scale={y:1})
		#optimizer=tf.train.AdamOptimizer(1e-1)
		#inference.initialize(optimizer)
		#inference.run(n_iter=10000)
		sess.run(tf.global_variables_initializer())

		for i in range(inference.n_iter):
			loss = inference.update({x:xdata,yph:ydata})
			print(loss['loss'])

		theta_estimate = sess.run(var)
		print('diff of thets',np.sum((theta_true-theta_estimate)**2))




