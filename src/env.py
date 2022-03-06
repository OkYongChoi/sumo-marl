import traci
from sumolib import checkBinary
import numpy as np


class TrafficEnv:
    def __init__(self, mode='binary'):
        # If the mode is 'gui', it renders the scenario.
        if mode == 'gui':
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')
        self.sumoCmd = [self.sumoBinary, "-c", './scenario/sample.sumocfg', '--no-step-log', '-W']

        self.time = None
        self.decision_time = 10
        self.n_intersections = 4
        self.n_phase = 2

    def reset(self):
        traci.start(self.sumoCmd)
        traci.simulationStep()
        self.time = 0

        return self.get_state()

    def get_state(self):
        # state: N X D array, where N is the number of traffic lights and D is the dimension of each observation
        state = []
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            observation = []
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                observation.append(traci.lane.getLastStepVehicleNumber(lane))
                observation.append(traci.lane.getLastStepHaltingNumber(lane))

            phase = [0 for _ in range(self.n_phase)]
            phase[traci.trafficlight.getPhase(intersection_ID)] = 1
            observation = np.array(observation + phase)
            state.append(observation)

        state = np.array(state)
        return state

    def apply_action(self, actions):
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            current_action = traci.trafficlight.getPhase(intersection_ID)
            if actions[i] == current_action:
                continue
            else:
                traci.trafficlight.setPhase(intersection_ID, actions[i])  # switch to next phase after yellow light

    def step(self, actions):
        self.apply_action(actions)
        for _ in range(self.decision_time):
            traci.simulationStep()
            self.time += 1

        state = self.get_state()
        reward = self.get_reward()
        done = self.get_done()
        return state, reward, done

    def get_reward(self):
        reward = [0.0 for _ in range(self.n_intersections)]
        for i, intersection_ID in enumerate(traci.trafficlight.getIDList()):
            for lane in traci.trafficlight.getControlledLanes(intersection_ID):
                reward[i] += traci.lane.getLastStepHaltingNumber(lane)

        reward = -np.array(reward)
        return reward

    def get_done(self):
        return traci.simulation.getMinExpectedNumber() == 0

    def close(self):
        traci.close()


if __name__ == "__main__":
    env = TrafficEnv()
    state = env.reset()
