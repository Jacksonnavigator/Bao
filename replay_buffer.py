import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []

    def push(self, game_examples):
        # game_examples: list of (state_tensor, policy_flat, value)
        for ex in game_examples:
            if len(self.buffer) >= self.capacity:
                # remove oldest
                self.buffer.pop(0)
            self.buffer.append(ex)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        # states are torch tensors (C,H,W)
        states = torch.stack(states)
        policies = torch.tensor(np.stack(policies), dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)
