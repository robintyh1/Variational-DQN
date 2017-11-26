import numpy as np
import sys
import math
import gym
from gym.utils import seeding
from gym import spaces

class NDeterministicChain(gym.Env):
	def __init__(self,N):
		self.T = N + 9
		self.N = N
		self.t = 0 
		self.position = 2
		self.features = np.zeros(N)

		self.high_state = np.ones(N)
		self.low_state = np.zeros(N)
		self.observation_space = spaces.Box(self.low_state,self.high_state)

		self.action_space = spaces.Discrete(2)

		self.env_spec = gym.spec(str(self.N)+'NDeterministicChain-v0')
		self.env_spec.timestep_limit = self.T

	def _seed(self,seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _compute_feature(self,pos):
		return np.append(np.ones(pos),np.zeros(self.N-pos))

	def _reset(self):
		self.t = 0
		self.position = 2
		self.features = self._compute_feature(self.position)
		return self.features.copy()

	def _step(self,action):
		self.t += 1
		if self.t == self.T:
			done = True
		else:
			done = False

		if self.position == 1:
			# in absorbing state
			self.features = self._compute_feature(self.position)
			reward = 1e-3
			return self.features.copy(),reward,done,{}
		elif self.position == self.N:
			# in absorbing state
			self.features = self._compute_feature(self.position)
			reward = 1.0
			return self.features.copy(),reward,done,{}
		else:
			if action == 0:
				# transition to left
				self.position -= 1
				self.features = self._compute_feature(self.position)
				reward = 0
				return self.features.copy(),reward,done,{}
			elif action == 1:
				# transition to right
				self.position += 1
				self.features = self._compute_feature(self.position)
				reward = 0
				return self.features.copy(),reward,done,{}
			else:
				raise ValueError("action should be 0 or 1")
				
	def _render(self,mode='human',close=False):
		pass


