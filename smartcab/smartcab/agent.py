import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.total_reward = 0

        #Q matrix
        self.Q = {}

        # initial Q values
        self.initial_q = 30

        # Dictionary of states to be modeled
        self.states = {'no left turn' : 0, 'no right turn' : 1, 'stop' : 2, 'right turn' : 3, None : 4, 'forward' : 5, 'left' : 6, 'right' : 7}

        # Dictionary of possible actions to index states from Q matrix
        self.actions = {None : 0, 'forward' : 1, 'left' : 2, 'right' : 3}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Check if this state is included in the Q matrix. If not, create a new row in the matrix.
        if self.state not in self.Q:
            self.Q[self.state] = [self.initial_q for i  in range(len(self.env.valid_actions))]

        # TODO: Select action according to your policy
        learning_rate = 0.8
        action = self.env.valid_actions[np.argmax(self.Q[self.state])]
        discount = 0
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward
        # TODO: Learn policy based on state, action, reward

        # get the future state
        future_state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Update the Q matrix (add consideration of future reward)
        #self.Q[self.state][self.actions[action]] = self.Q[self.state][self.actions[action]] * (1 - learning_rate) + reward * learning_rate
        self.Q[self.state][self.actions[action]] = self.Q[self.state][self.actions[action]] + learning_rate * (reward + discount * self.Q[future_state][self.actions[action]] - self.Q[self.state][self.actions[action]])


        #print self.Q
        #self.total_reward += reward
        print self.total_reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
