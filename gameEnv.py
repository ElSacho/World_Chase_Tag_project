

# Define the environment class
class GameEnv:
    def __init__(self):
        self.game_state = GameState(initial_state)
    
    # Reset the environment to its initial state
    def reset(self):
        self.game_state = GameState(initial_state)
        return self.game_state.get_state()

    # Step the environment by taking an action
    def step(self, action):
        next_state = self.game_state.take_action(action)
        done = self.game_state.is_done()
        reward = self.game_state.get_reward()
        self.game_state = next_state
        return next_state, reward, done, {}

