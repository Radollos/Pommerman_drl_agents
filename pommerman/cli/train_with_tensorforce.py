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
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent
import pommerman.agents.MyAgents as myagents

import numpy as np
import threading


CLIENT = docker.from_env()
learing_episodes = 150000
testing_episodes = 1000


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, action):
        if self.visualize:
            self.gym.render()
        #print("[DEBUG] Action FROM agent: " + str(action))
        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        #print("[DEBUG] Reward TO agent: " + str(agent_reward))
        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        return agent_obs


def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="myagents::ppo,test::agents.SimpleAgent,"
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
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent or type(agent) == myagents.BaselineAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    agent = training_agent.initialize(env)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)

    print(
        "[INFO] Starting learning for {test_ep} timesteps".format(test_ep=learing_episodes)
    )

    runner = Runner(agent=agent, environment=wrapped_env)
    if learing_episodes > 0:
        runner.run(num_episodes=learing_episodes, max_episode_timesteps=2000, episode_finished=episode_finished_learning)

    print(
        "[INFO] Learning finished. Testing for {test_ep} games".format(test_ep=testing_episodes)
    )

    save_path = "./models/" + training_agent.algorithm + "/" + time.time()
    agent.save_model(save_path)

    rows = zip(runner.episode_timesteps, runner.episode_rewards)
    import csv
    with open(str(save_path) + "complete_results.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    #####
    # If testing flag is True, episode counter in Runner is updated only once
    # (gets reset to starting value from agent's counter on a start of every episode)
    # Thus testing never ends.
    # Workaround: invoke run for 1 episode N times
    # TODO: find better workaround
    #####

    for i in range(0, testing_episodes):
        runner.run(num_episodes=1, max_episode_timesteps=2000, episode_finished=episode_finished_testing, testing=True, deterministic=True)

    # print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
    #       runner.episode_times
    #       )

    print(
        "[INFO] Testing results: Won {rews}/{test_ep}".format(test_ep=testing_episodes, rews=runner.episode_rewards[-testing_episodes:].count(1))
    )
    print(
        "[INFO] Testing results: Average timesteps: {times}".format(test_ep=testing_episodes, times=np.mean(runner.episode_timesteps[-testing_episodes:]))
    )
    try:
        runner.close()
    except AttributeError as e:
        pass


def episode_finished_learning(runner):
    log_interval = 100
    if runner.episode % log_interval == 0:
        print(
            "[LOG] {ep}/{total_ep}, {global_timestep} | Average of last {log_interval} episodes: rewards: {rew}, timesteps {tss}"
                .format(ep=runner.episode,
                        total_ep=learing_episodes,
                        global_timestep=runner.global_timestep,
                        log_interval=log_interval,
                        rew=np.mean(runner.episode_rewards[-log_interval:]),
                        tss=np.mean(runner.episode_timesteps[-log_interval:])
                        )
        )
        #print("[LOG] Episode {ep} finished after {ts} timesteps (reward: {reward})".format(ep=runner.episode, ts=runner.episode_timestep,
         #                                                                        reward=runner.episode_rewards[-1]))
    return True


def episode_finished_testing(runner):
    print("[LOG] Testing episode finished: reward: {reward}, {ts} timesteps".format(ep=runner.episode, ts=runner.episode_timestep, reward=runner.episode_rewards[-1]))
    return True


if __name__ == "__main__":
    main()
    threading.enumerate()
