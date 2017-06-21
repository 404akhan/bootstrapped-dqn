import itertools
import os
import time
import argparse
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys
import random

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

from ptan.common import runfile, env_params, utils, wrappers
from ptan.actions.epsilon_greedy import ActionSelectorEpsilonGreedy
from ptan import experience, agent

from lib import plotting
from collections import deque, namedtuple

def make_env():
    env_spec = gym.spec('ppaquette/DoomBasic-v0')
    env_spec.id = 'DoomBasic-v0'
    env = env_spec.make()
    e = wrappers.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=80, height=80, grayscale=True)

    return e
env = make_env()

from gym import wrappers
env = wrappers.Monitor(env, 'docs/monitor')

NOOP, SHOOT, RIGHT, LEFT = 0, 1, 2, 3
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor():
    def __init__(self):
        pass

    def process(self, sess, state):
        return np.squeeze(state)

class Estimator():
    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        self.X_pl = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = self.X_pl
        batch_size = tf.shape(self.X_pl)[0]

        conv1 = tf.contrib.layers.conv2d(
            X, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv1")
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv2 = tf.contrib.layers.conv2d(
            pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv2")
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv3 = tf.contrib.layers.conv2d(
            pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv3")
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        flattened = tf.contrib.layers.flatten(pool3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 64)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS), activation_fn=None)

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        global_step, _, loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    saver,
                    num_episodes,
                    replay_memory_size=10000,
                    replay_memory_init_size=1000,
                    update_target_estimator_every=500,
                    epsilon_decay_steps=6000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.,
                    batch_size=32):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    for i_episode in range(num_episodes):

        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        total_reward = 0
        actions_tracker = [0, 1, 2, 3]

        for t in itertools.count():

            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
                saver.save(sess, 'docs/dqn-doom-basic', global_step=total_t)

            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            actions_tracker.append(action)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))   
            total_reward += reward

            if total_t > replay_memory_init_size:
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            state = next_state
            total_t += 1

            if done:
                print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end=", ")
                print('reward %f, steps %d, eps %f' % (total_reward, t, epsilon), end=", ")
                counts_action = np.bincount(actions_tracker) - 1
                print('noop %d, shoot %d, right %d, left %d' % \
                    (counts_action[0], counts_action[1], counts_action[2], counts_action[3]))
                break

tf.reset_default_graph()

global_step = tf.Variable(0, name='global_step', trainable=False)
    
q_estimator = Estimator(scope="q")
target_estimator = Estimator(scope="target_q")

state_processor = StateProcessor()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
                    env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    state_processor=state_processor,
                    saver=saver,
                    num_episodes=1000)

    env.close()
    gym.upload('docs/monitor', api_key='sk_LsDbvQXoSvqD1z4OsVjqeQ')
