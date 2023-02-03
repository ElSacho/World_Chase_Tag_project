# Define the agent class
class Agent:
    def __init__(self, env):
        self.env = env

    # Choose an action based on the current state
    def choose_action(self, state):
        pass

    # Learn from the experience
    def learn(self, state, action, next_state, reward, done):
        pass
