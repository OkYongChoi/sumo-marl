import os
import sys
import argparse
import matplotlib.pyplot as plt

from env import TrafficEnv
from maddpg import MADDPG
from utils import get_average_travel_time


parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true",
                    help= "whether render while training or not")
args = parser.parse_args()

if __name__ == "__main__":

    # Before the start, should check SUMO_HOME is in your environment variables
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # configuration
    state_dim = 10
    action_dim = 2
    n_agents = 4
    n_episode = 100

    # Create an Environment and RL Agent
    env = TrafficEnv("gui") if args.render else TrafficEnv()
    agent = MADDPG(n_agents, state_dim, action_dim)

    # Train your RL agent
    performance_list = []
    for episode in range(n_episode):

        state = env.reset()
        reward_epi = []
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False

        while not done:
            # select action according to a given state
            for i in range(n_agents):
                action, action_prob = agent.select_action(state[i, :], i)
                actions[i] = action
                action_probs[i] = action_prob

            # apply action and get next state and reward
            before_state = state
            state, reward, done = env.step(actions)

            # make a transition and save to replay memory
            transition = [before_state, action_probs, state, reward, done]
            agent.push(transition)

            # train an agent
            if agent.train_start():
                for i in range(n_agents):
                    agent.train_model(i)
                agent.update_eps()

            if done:
                break

        env.close()
        average_traveling_time = get_average_travel_time()
        performance_list.append(average_traveling_time)

        print(f"Episode: {episode+1}\t Average Traveling Time:{average_traveling_time}\t Eps:{agent.eps}")

    # Save the model
    agent.save_model("results/trained_model.th")

    # Plot the performance
    plt.style.use('ggplot')
    plt.figure(figsize=(10.8, 7.2), dpi=120)
    plt.plot(performance_list)
    plt.xlabel('# of Episodes')
    plt.ylabel('Average Traveling Time')
    plt.title('Performance of MADDPG for controlling traffic lights')
    plt.savefig('./results/performance.png')