# Variational DQN
The implementation of Variational DQN based on Chainer, Tensorflow and Edward. Part of the Chainer code is borrowed from Chainer tutorial on DQN.

Variational DQN leverages variational inference subroutines to update DQN parameters.

### Use the code
To run Variational DQN or DQN on Cartpole for 200 episodes
```
python main_VDQN.py --env CartPole-v1 --episodes 200
python main_DQN.py --env CartPole-v1 --episodes 200
```

### References

[Tang et al., *Variational Deep Q Network*. Bayesian Deep Learning Workshop, NIPS, 2017.][vdqn]

[Tang et al., *Exploration by Distributional Reinforcement Learning*.
International Joint Conference on Artificial Intelligence (IJCAI), 2018.][exp]

[vdqn]: https://arxiv.org/abs/1711.11225
[exp]: https://arxiv.org/abs/1805.01907


