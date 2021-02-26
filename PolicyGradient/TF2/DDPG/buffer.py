import numpy as np

class ReplayBuffer():
    """
    Here we will store all our state, action, reward, next state, terminal flag
    """
    def __init__(self, max_size, input_shape, n_actions):
        #n_actions here actually means the number of compounds to the action

        self.mem_size = max_size # the memory is bounded, we will overwrite when too full
        self.mem_cntr = 0 # counter
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        """Function to store the transition"""
        index = self.mem_cntr % self.mem_size # Determine position of first available memory

        # Here we save the transitions
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """Sample the buffer"""

        # Check how much of the memory we actually filled up
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


