import numpy as np
from collections import deque
import random

class ReplayBuffer(object):

	def __init__(self,buffersize):
		self.buffersize = buffersize
		self.buffer = deque()
		self.currentsize = 0

	def append(self,experience):
		'''
		experience: a list of tuple [(s_1,a_1,r_1,d_1,s_2)]
		'''
		for e in experience:
			self.buffer.append(e)
		self.currentsize += len(experience)

	def popleft(self):
		while self.currentsize > self.buffersize:
			self.buffer.popleft()
			self.currentsize -= 1

	def sample(self,batchsize):
		minibatch = random.sample(self.buffer,batchsize)
		batch_state = np.array([batch[0] for batch in minibatch])
		batch_action = np.array([batch[1] for batch in minibatch])
		batch_reward = np.array([batch[2] for batch in minibatch])
		batch_done = np.array([batch[3] for batch in minibatch])
		batch_nextstate = np.array([batch[4] for batch in minibatch])
		return batch_state,batch_action,batch_reward,batch_done,batch_nextstate



test = False
if test:
	import gym
	replaybuffer = ReplayBuffer(5)
	env = gym.make('CartPole-v0')
	obs = env.reset()
	done = False
	while not done:
		action = env.action_space.sample()
		nextobs,r,done,_ = env.step(action)
		replaybuffer.append([(obs,action,r,nextobs)])
		replaybuffer.popleft()
		print(replaybuffer.currentsize)
		if done:
			break
	obs,actions,reward,done,nextobs = replaybuffer.sample(3)
	print(obs.shape)
	print(actions.shape)
	print(reward.shape)
	print(done.shape)
	print(nextobs.shape)






