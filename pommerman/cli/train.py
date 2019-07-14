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
from pommerman.agents.MyAgents import BaselineAgent
from stable_baselines.common.vec_env import DummyVecEnv

CLIENT = docker.from_env()
learning_episodes = 10000000
testing_episodes = 100


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnvBaselines(gym.Env):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''
    metadata = {'render.modes': ['human']}

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

    training_player_lasts = {
        'ammo': 1,
        'blast': 2,
        'kick': 0
    }

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

        self.training_player_lasts = {
            'ammo': 1,
            'blast_strength': 2,
            'can_kick': 0,
            'alive': 4
        }

    def step(self, action):
        if self.visualize:
            self.gym.render()

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        # if agent_reward != -1.0 or agent_reward != 1.0:
        #     agent_reward = self._shape_reward(agent_reward, state[self.gym.training_agent], action)

        #print("[DEBUG] Reward TO agent: " + str(agent_reward) + ", terminal: " + str(terminal))

        self.actions_freq[action] += 1
        self.total_timestep_counter += 1
        self.episode_timestep_counter += 1
        if terminal:
            self.episode_counter += 1
            self.episode_rewards.append(agent_reward)
            self.episode_timesteps.append(self.episode_timestep_counter)
            self.episode_timestep_counter = 0


        # agent_state = np.where((agent_state > 10), 11, agent_state)
        agent_state = np.true_divide(agent_state, WrappedEnvBaselines.max_obs_board)
        # agent_state = agent_state[:-4]
        return agent_state, agent_reward, terminal, info

    def reset(self):
        # self.training_player_lasts = {
        #     'ammo': 1,
        #     'blast_strength': 2,
        #     'can_kick': 0,
        #     'alive': 4
        # }
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[self.gym.training_agent])
        # agent_obs = np.where((agent_obs > 10), 11, agent_obs)
        res = np.true_divide(agent_obs, WrappedEnvBaselines.max_obs_board)
        # res = res[:-4]
        return res

    def _shape_reward(self, current_reward, agent_state, action):
        ### Training agent manual reward ###
        # Reward for picking up blast_range upgrade
        if agent_state['blast_strength'] > self.training_player_lasts['blast_strength']:
            current_reward += 0.1
            self.training_player_lasts['blast_strength'] = agent_state['blast_strength']
        # Reward for picking up can_kick upgrade
        if self.training_player_lasts['can_kick'] == 0 and agent_state['can_kick'] == 1:
            current_reward += 0.1
            self.training_player_lasts['can_kick'] = 1
        # No way to clearly determine when blast_range update is picked up

        # Reward for dying enemies and penalty for losing life or ally
        # current_reward += (4-len(agent_state['alive'])) * 0.1


        # if len(agent_state['alive']) < 4:
        #     # Player check
        #     if not 10 in agent_state['alive']:
        #         current_reward -= 0.25
        #     # Enemy 1 check
        #     if not 11 in agent_state['alive']:
        #         current_reward += 0.15
        #     # Enemy 2 check
        #     if not 12 in agent_state['alive']:
        #         current_reward -= 0.10
        #     # Ally check
        #     if not 13 in agent_state['alive']:
        #         current_reward += 0.15
        player_position = agent_state['position']
        if action == 5:
            # Reward for placing bomb, used to encourage aggresive play
            # Exp: if bomb is at the same position as player then he must have placed it,
            # if bomb timer is at max(9) it was JUST placed
            if agent_state['bomb_life'][player_position[0], player_position[1]] >= 9:
                current_reward += 0.2
            # Penalty for trying to place bomb when out of ammo
            # else:
            #     current_reward -= 0.01

        # enemies_positions = np.where((agent_state['board'] > 10) & (agent_state['board'] < 14))
        # enemies_positions = list(zip(enemies_positions[0], enemies_positions[1]))
        # if len(enemies_positions) > 0:
        #     max_dist = min([abs(player_position[0] - position[0]) + abs(player_position[1] - position[1]) for position in enemies_positions])
        #     current_reward -= max_dist * 0.01

        return current_reward

    def render(self, mode='human', close=False):
        if self.visualize:
            self.gym.render()


def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetitionFast-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="myagents::ppo2,test::agents.SimpleAgent,"
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
        default=False,
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
    atexit.register(functools.partial(clean_up_agents, agents))

    env = make(config, agents, game_state_file)
    # env.observation_space.low = env.observation_space.low[:-4]
    # env.observation_space.high = env.observation_space.high[:-4]
    # env.observation_space.shape = (env.observation_space.shape[0] - 4,)
    training_agent = None
    # training_agent = agents[1]
    # env.set_training_agent(training_agent.agent_id)
    for agent in agents:
        if type(agent) == BaselineAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    wrapped_env = WrappedEnvBaselines(env, visualize=args.render)
    vecenv = DummyVecEnv([lambda: wrapped_env])
    agent = training_agent.initialize(vecenv)
    #agents[0].initialize(wrapped_env.gym, directory="C:\\Studia\\Magisterka\\playground\\pommerman\\cli\\models\ppo2\\1557659082.7873366")

    print(
        "[INFO] Starting learning for {test_ep} episodes".format(test_ep=learning_episodes)
    )

    agent.learn(total_timesteps=learning_episodes)

    print(
        "[INFO] Learning finished. Done {eps} episodes. Testing for {test_ep} games".format(test_ep=testing_episodes, eps=wrapped_env.episode_counter)
    )

    save_path = "./models/" + training_agent.algorithm + "/" + str(time.time())
    try:
        os.mkdir("./models/" + training_agent.algorithm + "/")
    except:
        pass
    agent.save(save_path)

    rows = zip(wrapped_env.episode_timesteps, wrapped_env.episode_rewards)
    import csv
    with open(str(save_path) + "complete_results.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    observed_state = wrapped_env.reset()
    test_finished_episodes = 0
    test_won_episodes = 0
    test_total_timesteps = 0
    while test_finished_episodes < testing_episodes:
        test_total_timesteps += 1
        action, _state = agent.predict(observed_state)
        observed_state, reward, episode_finished, info = wrapped_env.step(action)
        wrapped_env.render()
        if episode_finished:
            test_finished_episodes += 1
            if reward == 1:
                test_won_episodes += 1
            observed_state = wrapped_env.reset()
        if testing_episodes - test_finished_episodes < 5:
            wrapped_env.visualize = True # visualizing all tests takes too long

    print(
        "[INFO] Testing results: Won {rews}/{test_ep}".format(test_ep=testing_episodes, rews=test_won_episodes)
    )
    print(
        "[INFO] Testing results: Average timesteps: {times}".format(test_ep=testing_episodes, times=float(test_total_timesteps)/testing_episodes)
    )


if __name__ == "__main__":
    main()
