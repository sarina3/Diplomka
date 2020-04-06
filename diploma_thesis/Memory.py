import random

class ExperienceReplay():
    def __init__(self, memory_size):
        self.memory = []
        self.max_size = memory_size
        self.index = 0

    def push(self, data):
        if self.__len__() < self.max_size:
            self.memory.append(data)
            self.index = (self.index + 1 ) % self.max_size
        else:
            self.memory[self.index] = data
            self.index = (self.index + 1 ) % self.max_size

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


