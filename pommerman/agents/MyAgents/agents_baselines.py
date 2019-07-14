from collections import defaultdict
import queue
import random
import tensorflow as tf
import numpy as np

from pommerman.agents import BaseAgent
from pommerman import constants
from pommerman import utility
from pommerman import characters
from tensorforce.agents import DQNAgent, PPOAgent

from stable_baselines import DQN, PPO2, A2C
from stable_baselines.deepq.policies import LnMlpPolicy, MlpPolicy, DQNPolicy, FeedForwardPolicy
from stable_baselines.common import policies

max_obs_board = [13 for i in range(121)]  # board state
max_obs_board.extend([10 for i in range(121)])  # bomb blast strengths
max_obs_board.extend([9 for i in range(121)])  # bomb timers
max_obs_board.extend([10, 10])  # position of player
max_obs_board.extend([5])  # player ammo
max_obs_board.extend([10])  # player blast strength
max_obs_board.extend([1])  # player can kick
max_obs_board.extend([13])  # teammate
max_obs_board.extend([13, 13, 13])  # enemies
max_obs_board = np.array(max_obs_board)


class BaselineAgent(BaseAgent):

    def __init__(self, character=characters.Bomber, algorithm='ppo2'):
        super(BaselineAgent, self).__init__(character)
        self.algorithm = algorithm
        self.agent = None

    def act(self, obs, action_space):
        agent_state = self.gym.featurize(obs)
        agent_state = np.true_divide(agent_state, max_obs_board)
        return self.agent.predict(agent_state, deterministic=True)[0]

    def initialize(self, env, directory=None):
        agent = None
        self.gym = env
        if directory:
            print("[INFO] Restoring model from {dir}".format(dir=directory))

        if self.algorithm == "dddqn":
            if directory:
                agent = DQN.load(directory)
            else:
                agent = DDDQNAgentWrapper(env)
        if self.algorithm == "ppo2":
            if directory:
                agent = PPO2.load(directory)
            else:
                agent = PPO2AgentWrapper(env)
        if self.algorithm == "a2c":
            if directory:
                agent = A2C.load(directory)
            else:
                agent = A2CAgentWrapper(env)
        self.agent = agent
        return agent


class DQNAgentWrapper(DQNAgent):
    def __init__(self, actions, states):
        print("[INFO] Initializing DDQNAgent")

        super(DQNAgentWrapper, self).__init__(
                states=states,
                actions=actions,
                network=[
                    dict(type='dense', size=1024, activation='relu'),
                    dict(type='dense', size=256, activation='relu'),
                ],
                memory=dict(type='prioritized_replay', include_next_states=True, capacity=2000000, buffer_size=50000),
                actions_exploration=dict(type='epsilon_decay', initial_epsilon=0.99, final_epsilon=0.01, timesteps=3000000)
                , batching_capacity=2000
                , double_q_model=True
                , discount=0.90
                , update_mode=dict(unit='timesteps', batch_size=5000, frequency=1000)
                #, summarizer=dict(directory="/summarizer/", labels=['states', 'actions', 'rewards'])
            )

    def act(self, states, deterministic=False, independent=False, fetch_tensors=None, buffered=True):
        action = super(DQNAgentWrapper, self).act(states, deterministic=deterministic, independent=independent, fetch_tensors=fetch_tensors, buffered=buffered)
        return action
    
    def observe(self, terminal, reward):
        super(DQNAgentWrapper, self).observe(terminal, reward)


class PPOAgentWrapper(PPOAgent):
    def __init__(self, actions, states):
        print("[INFO] Initializing PPOAgent")

        super(PPOAgentWrapper, self).__init__(
            states=states,
            actions=actions,
            network=[
                dict(type='dense', size=1024, activation='relu'),
                dict(type='dense', size=256, activation='relu'),
            ],
            actions_exploration=dict(type='epsilon_decay', initial_epsilon=0.99, final_epsilon=0.01, timesteps=10000000),
            memory=dict(type='prioritized_replay', include_next_states=False, capacity=200000, buffer_size=5000),
            batching_capacity=2000,
            step_optimizer=dict(type='adam', learning_rate=1e-4),
            discount=0.90
            # ,summarizer=dict(directory="/summarizer/", labels=['states', 'actions', 'rewards'])
        )

    def act(self, states, deterministic=False, independent=False, fetch_tensors=None, buffered=True):
        action = super(PPOAgentWrapper, self).act(states, deterministic=deterministic, independent=independent,
                                                  fetch_tensors=fetch_tensors, buffered=buffered)
        return action

    def observe(self, terminal, reward):
        super(PPOAgentWrapper, self).observe(terminal, reward)


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                           feature_extraction="mlp",
                                           obs_phs=obs_phs,
                                           net_arch=[dict(vf=[1024, 256], pi=[1024, 256])],
                                           layer_norm=True,
                                           dueling=True,
                                           **_kwargs)


class DDDQNAgentWrapper(DQN):
    def __init__(self, env):
        print("[INFO] Initializing DDDQN with PER agent")
        policy_kwargs = dict(act_fun=tf.nn.relu, layers=[700, 350])

        DDDQNAgentWrapper.instance = super(DDDQNAgentWrapper, self).__init__(
            policy=LnMlpPolicy, ### dueling inside policy
            policy_kwargs=policy_kwargs,
            env=env,
            buffer_size=200000,
            batch_size=2000,
            gamma=0.99,
            exploration_fraction=0.90,
            #exploration_final_eps=0.01,
            prioritized_replay=True, ### prioritized replay
            param_noise=True, ### parameter noise
            target_network_update_freq=1000, ### double
            checkpoint_freq=500000,
            verbose=1,
            tensorboard_log="./tensorboard/dqn/"
        )


class PPO2AgentWrapper(PPO2):
    def __init__(self, env):
        print("[INFO] Initializing PPO2 agent")
        policy_kwargs = dict(act_fun=tf.nn.relu, layers=[700, 350])

        super(PPO2AgentWrapper, self).__init__(
            policy=policies.MlpPolicy,  ### dueling inside policy
            policy_kwargs=policy_kwargs,
            env=env,
            n_steps=2000,
            gamma=0.99,
            #cliprange=0.3,
            verbose=1,
            #noptepochs=2,
            #nminibatches=2,
            #ent_coef=0.03,
            tensorboard_log="./tensorboard/ppo/"
        )

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return super(PPO2AgentWrapper, self).predict(observation, state, mask, deterministic)


class A2CAgentWrapper(A2C):
    def __init__(self, env):
        print("[INFO] Initializing A2C agent")
        policy_kwargs = dict(act_fun=tf.nn.relu, layers=[700, 350])

        super(A2CAgentWrapper, self).__init__(
            policy=policies.MlpPolicy,
            policy_kwargs=policy_kwargs,
            env=env,
            n_steps=2000,
            gamma=0.99,
            verbose=1,
            #ent_coef=0.03,
            tensorboard_log="./tensorboard/a2c/"
        )
