import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def add(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))
    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)
    def __len__(self):
        return len(self.memory)
