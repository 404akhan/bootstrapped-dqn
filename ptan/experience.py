import torch
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque, OrderedDict

from .common import env_params

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments
    
    Every experience contains n+1 list of Experience entries
    """
    def __init__(self, env, agent, steps_count=1):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        """
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.total_rewards = []

    def __iter__(self):
        states, histories, cur_rewards = [], [], []
        for env in self.pool:
            states.append(env.reset())
            histories.append(deque())
            cur_rewards.append(0.0)

        while True:
            actions = self.agent(np.array(states))

            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx][0]
                history = histories[idx]
                next_state, r, is_done, _ = env.step(action)
                cur_rewards[idx] += r
                history.append(Experience(state=state, action=action, reward=r, done=is_done))
                while len(history) > self.steps_count+1:
                    history.popleft()
                if len(history) == self.steps_count+1:
                    yield tuple(history)
                states[idx] = next_state
                if is_done:
                    if len(history) > self.steps_count+1:
                        history.popleft()
                    # generate tail of history
                    while len(history) >= 1:
                        yield tuple(history)
                        history.popleft()
                    self.total_rewards.append(cur_rewards[idx])
                    cur_rewards[idx] = 0.0
                    states[idx] = env.reset()
                    history.clear()

    def pop_total_rewards(self):
        r = self.total_rewards
        self.total_rewards = []
        return r


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size=None):
        self.buffer_size = buffer_size
        self.experience_source = experience_source
        self.experience_source_iter = iter(experience_source)
        self.buffer = deque()

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size: 
        :return: 
        """
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        keys = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
        return [self.buffer[key] for key in keys]

    def batches(self, batch_size):
        """
        Iterate batches of given size once (i.e. one epoch over buffer)
        :param batch_size: 
        """
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        while samples > 0:
            entry = next(self.experience_source_iter)
            self.buffer.append(entry)
            samples -= 1
        if self.buffer_size is not None:
            while len(self.buffer) > self.buffer_size:
                self.buffer.popleft()


class PrioritizedReplayBuffer:
    def __init__(self, experience_source, buffer_size, prob_alpha=1.0, weight_beta=1.0):
        """
        Construct prioritized replay buffer
        :param experience_source: source of experience we're going to use 
        :param buffer_size: max size of buffer 
        :param prob_alpha: exponent for probabilities 
        :param weight_beta: exponent for weights 
        """
        self.buffer_size = buffer_size
        self.experience_source = experience_source
        self.experience_source_iter = iter(experience_source)
        self.prob_alpha = prob_alpha
        self.weight_beta = weight_beta
        self.buffer = deque()
        self.probs = deque()

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        """
        Sample batch from experience replay, returning data, indices and sample weights.
        
        Indices should be passed for call update_priorities()
        :param batch_size: 
        :return: tuple of (batch_data, batch_indices, weights) 
        """
        indices = range(len(self.buffer))
        batch_idx = np.random.choice(indices, batch_size, p=self.probs)
        batch_dat = [self.buffer[idx] for idx in batch_idx]
        weights = [(1. / (len(self.buffer) * self.probs[idx])) ** self.weight_beta for idx in batch_idx]

        # normalize weights
        max_w = max(weights)
        res_weights = [w / max_w for w in weights]
        return batch_dat, batch_idx, res_weights

    def populate(self, samples):
        """
        Fetch given amount of samples into the buffer
        :param samples: 
        """
        max_prob = max(self.probs) if self.probs else 1.0

        while samples > 0:
            entry = next(self.experience_source_iter)
            self.buffer.append(entry)
            self.probs.append(max_prob)
            samples -= 1
        while len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
            self.probs.popleft()
        self._normalize_probs()

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update batch item priorities, should be called after TD error 
        calculation to update probability of sampling.
        :param batch_indices: indices returned by sample() call 
        :param batch_priorities: list of numbers reflecting priorities 
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.probs[idx] = priority ** self.prob_alpha
        self._normalize_probs()

    def _normalize_probs(self):
        s = sum(self.probs)
        for idx, p in enumerate(self.probs):
            self.probs[idx] = p / s


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience 
    batch to form suitable to learning.
    """
    def preprocess(self, batch):
        raise NotImplementedError


class QLearningPreprocessor(BatchPreprocessor):
    """
    Supports SimpleDQN, TargetDQN, DoubleDQN and can additionally feed TD-error back to 
    experience replay buffer.
    
    To use different modes, use appropriate class method
    """
    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states_first, states_last):
        """
        Calculates apropriate q values for first and last states. Way of calculate depends on our settings.
        :param states_first: numpy array of first states 
        :param states_last: numpy array of last states 
        :return: tuple of numpy arrays of q values 
        """
        # here we need both first and last values calculated using our main model, so we
        # combine both states into one batch for efficiency and separate results later
        if self.target_model is None or self.use_double_dqn:
            states_t = torch.from_numpy(np.concatenate((states_first, states_last), axis=0))
            states_v = Variable(states_t)
            if env_params.get().cuda_enabled:
                states_v = states_v.cuda()
            res_both = self.model(states_v).data.cpu().numpy()
            return res_both[:len(states_first)], res_both[len(states_first):]

        # in this case we have target_model set and use_double_dqn==False
        # so, we should calculate first_q and last_q using different models
        states_first_v = Variable(torch.from_numpy(states_first))
        states_last_v = Variable(torch.from_numpy(states_last))
        if env_params.get().cuda_enabled:
            states_first_v = states_first_v.cuda()
            states_last_v = states_last_v.cuda()
        q_first = self.model(states_first_v).data
        q_last = self.target_model(states_last_v).data
        return q_first.cpu().numpy(), q_last.cpu().numpy()

    def _calc_target_rewards(self, states_last, q_last):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states_last: numpy array of last states from the games
        :param q_last: numpy array of last q values 
        :return: vector of target rewards 
        """
        # in this case we handle both simple DQN and target DQN
        if self.target_model is None or not self.use_double_dqn:
            return q_last.max(axis=1)

        # here we have target_model set and use_double_dqn==True
        actions = q_last.argmax(axis=1)
        # calculate Q values using target net
        states_last_v = Variable(torch.from_numpy(states_last))
        if env_params.get().cuda_enabled:
            states_last_v = states_last_v.cuda()
        q_last_target = self.target_model(states_last_v).data.cpu().numpy()
        return q_last_target[range(q_last_target.shape[0]), actions]

    def preprocess(self, batch):
        """
        Calculates data for Q learning from batch of observations
        :param batch: list of lists of Experience objects
        :return: tuple of numpy arrays: 
            1. states -- observations
            2. target Q-values
            3. vector of td errors for every batch entry
        """
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0, qL = self._calc_Q(state_0, state_L)
        rewards = self._calc_target_rewards(state_L, qL)

        td = np.zeros(shape=(len(batch),))

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final reward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward and calculate td error
            act = exps[0].action
            td[idx] = q0[idx][act] - total_reward
            q0[idx][act] = total_reward

        return state_0, q0, td
