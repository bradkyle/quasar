from agents.agent import Agent

class DistralAgent(Agent):
    def __init__(self,scope,observation_shape,rewards_shape,candidates_shape,memory):
        Agent.__init__(self,scope,observation_shape,rewards_shape,candidates_shape,memory)