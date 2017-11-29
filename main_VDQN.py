"""
Implement Variational DQN in Edward and Tensorflow
"""

import tensorflow as tf
import numpy as np
import gym
import os
import argparse

from models_VDQN import VariationalQNetwork, update_target, NoiseSampler
from utils import ReplayBuffer

# import HardMDP


def main():

    parser = argparse.ArgumentParser(description='VDQN')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Name of the OpenAI Gym environment')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--target-update-period', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=.99)
    args = parser.parse_args()

    # HYPERPARAMETERS
    episodes = args.episodes
    envname = args.env

    seed = args.seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    hiddendict = [100, 100]
    batchsize = 64
    buffersize = 1000000
    initialsize = 500
    target_update_period = args.target_update_period
    lr_VI = args.lr
    gamma = args.gamma
    totalstep = 0
    reward_scale = 1

    # MAIN ITERATIONS

    logdir = (args.logdir + 'VDQN/' + envname + '/lr_' +
              str(args.lr) + '_episodes' + str(args.episodes))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with tf.Session() as sess:

        # INITIALIZATION
        env = gym.make(envname)
        obssize = env.observation_space.low.size
        actsize = env.action_space.n
        replaybuffer = ReplayBuffer(buffersize)
        optimizer = tf.train.AdamOptimizer(lr_VI)
        Qactionnet = VariationalQNetwork(obssize, actsize,
                                         hiddendict, sess=sess,
                                         scope='principle',
                                         optimizer=optimizer)
        Qtargetnet = VariationalQNetwork(obssize, actsize,
                                         hiddendict, sess=sess, scope='target')
        noisesampler = NoiseSampler(Qactionnet.Wshape, Qactionnet.bshape)

        sess.run(tf.global_variables_initializer())

        update_target(Qtargetnet, Qactionnet)

        # RECORD
        VIlossrecord = []
        Bellmanlossrecord = []
        rewardrecord = []

        # ITERATIONS
        for episode in range(episodes):

            # start
            obs = env.reset()
            done = False

            rsum = 0
            while not done:
                # sample a noise and compute
                Wnoise, bnoise = noisesampler.sample(1)
                # compuet Q value
                Qvalue = Qactionnet.compute_Qvalue(obs[None], Wnoise, bnoise)
                # select action
                action = np.argmax(Qvalue.flatten())
                # step
                nextobs, reward, done, _ = env.step(action)
                # record experience
                done_ = 1 if done else 0
                reward_ = reward * reward_scale
                experience = [(obs, action, reward_, done_, nextobs)]
                # append experience to buffer
                replaybuffer.append(experience)
                replaybuffer.popleft()
                # update
                obs = nextobs
                totalstep += 1
                rsum += reward

                if replaybuffer.currentsize >= initialsize:
                    # sample minibatch
                    (batch_obs, batch_act, batch_reward, batch_done,
                     batch_nextobs) = replaybuffer.sample(batchsize)
                    # sample noise for computing target
                    Wnoise, bnoise = noisesampler.sample(batchsize)
                    # compute target value
                    Qall = Qtargetnet.compute_Qvalue(batch_nextobs,
                                                     Wnoise, bnoise)
                    Qtarget = (gamma * np.max(Qall, axis=1) *
                               (1 - batch_done) + batch_reward)
                    # udpate principle network by VI
                    VIloss = Qactionnet.train_on_sample(batch_obs,
                                                        batch_act, Qtarget)
                    # comptue bellman error loss
                    # Wnoise_new, bnoise_new = noisesampler.sample(batchsize)
                    Wnoise_new, bnoise_new = Wnoise, bnoise
                    Qpred = Qactionnet.compute_Qvalue(batch_obs,
                                                      Wnoise_new, bnoise_new)
                    Qpredact = Qpred[np.arange(batchsize), batch_act]
                    Bellmanloss = np.mean((Qpredact - Qtarget)**2)
                    VIlossrecord.append(VIloss['loss'])
                    Bellmanlossrecord.append(Bellmanloss)

                if (totalstep + 1) % target_update_period == 0:
                    update_target(Qtargetnet, Qactionnet)
                    print("update target")

                if done:
                    break

            # record
            rewardrecord.append(rsum)
            if len(VIlossrecord) > 10:
                meanVIloss = np.mean(VIlossrecord[-10:])
            else:
                meanVIloss = np.float('nan')
            if len(Bellmanlossrecord) > 10:
                meanbellmanloss = np.mean(Bellmanlossrecord[-10:])
            else:
                meanbellmanloss = np.float('nan')
            meanreward = np.mean(rewardrecord[-10:])

            print("episode %d buffer size %d meanVIloss %f"
                  " meanbellmanloss %f meanreward %f"
                  % (episode, replaybuffer.currentsize,
                     meanVIloss, meanbellmanloss, meanreward))

            if (1 + episode) % 5 == 0:
                np.save(logdir + '/VIloss_' +
                        str(seed), VIlossrecord)
                np.save(logdir + '/bellmanloss_' +
                        str(seed), Bellmanlossrecord)
                np.save(logdir + '/reward_' +
                        str(seed), rewardrecord)


if __name__ == '__main__':
    main()
