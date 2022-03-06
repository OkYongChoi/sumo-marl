import os
import sys

from env import TrafficEnv
from maddpg import MADDPG
from utils import get_average_travel_time

if __name__ == "__main__":

    # Before the start, you should check SUMO_HOME is in your environment variables
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # Hyperparameters
    state_dim = 10
    action_dim = 2
    n_agents = 4

    # Create an Environment and RL Agent
    env = TrafficEnv('gui')
    agent = MADDPG(n_agents, state_dim, action_dim)

    # Load your trained RL Agent
    agent.load_model("results/trained_model.th")
    agent.eps = 0.0

    # Visualize your RL agent
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

        if done:
            break

    env.close()
    print(get_average_travel_time())
