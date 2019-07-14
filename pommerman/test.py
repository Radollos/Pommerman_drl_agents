"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os
import time

import argparse
import docker
import gym
import numpy as np
import itertools
from gym import spaces

from pommerman import helpers, make
from pommerman import agents as agnts
#from pommerman.agents import BaselineAgent

CLIENT = docker.from_env()
testing_episodes = 10

def clean_up_agents(agents):
    return [agent.shutdown() for agent in agents]


class WrappedEnvBaselines(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    ### Used to normalize observed state vector
    ### Observed state vector has different ranges across its values
    max_obs_board = [13 for i in range(121)] # board state
    max_obs_board.extend([10 for i in range(121)]) # bomb blast strengths
    max_obs_board.extend([9 for i in range(121)]) # bomb timers
    max_obs_board.extend([10, 10]) # position of player
    max_obs_board.extend([5]) # player ammo
    max_obs_board.extend([10]) # player blast strength
    max_obs_board.extend([1]) # player can kick
    max_obs_board.extend([13]) # teammate
    max_obs_board.extend([13, 13, 13])  # enemies
    max_obs_board = np.array(max_obs_board)

    def __init__(self, gym, visualize=False):
        super(WrappedEnvBaselines, self).__init__()
        self.gym = gym
        self.visualize = visualize
        self.action_space = gym.action_space
        self.observation_space = gym.observation_space

        self.total_timestep_counter = 0
        self.episode_counter = 1
        self.episode_rewards = []
        self.episode_timesteps = []
        self.episode_timestep_counter = 0
        self.actions_freq = [0, 0, 0, 0, 0, 0]

    def step(self, action):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        self.actions_freq[action] += 1
        self.total_timestep_counter += 1
        self.episode_timestep_counter += 1
        if terminal:
            self.episode_counter += 1
            self.episode_rewards.append(agent_reward)
            self.episode_timesteps.append(self.episode_timestep_counter)
            self.episode_timestep_counter = 0
        agent_state = np.true_divide(agent_state, WrappedEnvBaselines.max_obs_board)
        return agent_state, agent_reward, terminal, info

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[self.gym.training_agent])
        res = np.true_divide(agent_obs, WrappedEnvBaselines.max_obs_board)
        return res

    def render(self, mode='human', close=False):
        if self.visualize:
            return self.gym.render(mode)


def main():
    '''CLI interface to bootstrap testing'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo2,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=True,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    args = parser.parse_args()
    print(args)
    config = args.config
    game_state_file = args.game_state_file

    agents_string = args.agents.split(",")

    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(agents_string)
    ]

    env = make(config, agents, game_state_file)

    for agent in agents:
        if type(agent) == agnts.TensorForceAgent or type(agent) == agnts.BaselineAgent:
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    wrapped_env = WrappedEnvBaselines(env, visualize=args.render)
    atexit.register(functools.partial(clean_up_agents, agents))

    from stable_baselines import PPO2, DQN, A2C
    print(
        "[INFO] Loading PPO model"
    )
    #models/ppo2/1558920269.0932562
    agent = PPO2.load("pommerman/cli/models/1559359177.602215")

    test_finished_episodes = 0
    test_won_episodes = 0
    test_total_timesteps = 0
    actions_freq = [0, 0, 0, 0, 0, 0]

    print(
        "[INFO] Starting testing for {test_length} games".format(test_length=testing_episodes)
    )

    observed_state = wrapped_env.reset()
    while test_finished_episodes < testing_episodes:

        test_total_timesteps += 1
        action, _states = agent.predict(observed_state)
        actions_freq[action] += 1
        observed_state, reward, episode_finished, info = wrapped_env.step(action)
        if test_total_timesteps == 1:
            time.sleep(1)
        if testing_episodes - test_finished_episodes <= 5:
            wrapped_env.visualize = True

        if episode_finished:
            test_finished_episodes += 1
            print("[LOG] Last episode reward: " + str(reward))
            if reward == 1:
                test_won_episodes += 1
            observed_state = wrapped_env.reset()
            time.sleep(1)

    print(
        "[INFO] Won episodes/total episodes: {rews}/{test_ep}".format(test_ep=testing_episodes, rews=test_won_episodes)
    )
    print(
        "[INFO] Average episode length: {times}".format(test_ep=testing_episodes, times=float(test_total_timesteps)/testing_episodes)
    )
    exit()


if __name__ == "__main__":
    main()
