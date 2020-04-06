import torch.nn as nn
import torch
import torch.optim as opt

from Neural_network import NeuralNetwork
from Memory import ExperienceReplay
from Statistics import Statistics
# from config import *
import random
import gym

learn_rate = 0.001
num_o_ep = 20000
gamma = 0.99
egreedy = 0.99
decay = 0.999
egreedy_final = 0.01
score_to_achieve = 200
report_interval = 10
solved = False
hidden_layer = 128
hidden_layer1 = 64
hidden_layer2 = 64
memory_size = 600000
batch_size = 64
update_target_frequency = 200
clip_err = False
double_DQN = True
load_weights = False
weights_file = 'weights_DQN.txt'
render_frequency = 100
weights_saving_frequency = 100
dueling = True

class Agent():
    def __init__(self,batch_size, env, device):
        self.env = env
        self.device = device
        self.num_o_input = env.observation_space.shape[0]
        self.num_o_output = env.action_space.n
        self.NN = NeuralNetwork(self.num_o_input,hidden_layer,hidden_layer1,hidden_layer2,self.num_o_output, dueling).to(device)
        self.targetNN = NeuralNetwork(self.num_o_input,hidden_layer,hidden_layer1,hidden_layer2,self.num_o_output, dueling).to(device)
        if load_weights:
            self.NN.load_state_dict(torch.load(weights_file))
            self.targetNN.load_state_dict(torch.load(weights_file))

        self.lossFunc = nn.MSELoss()
        self.optimizer = opt.Adam(params=self.NN.parameters(), lr=learn_rate)
        self.batch_size = batch_size
        self.memory = ExperienceReplay(memory_size)
        self.statistics = Statistics()
        self.egreedy = egreedy
        self.statistics.set('egreedy',self.egreedy)
        self.update_target_counter = 0

    def get_action(self,state):
        with torch.no_grad():
            state = torch.Tensor(state).to(self.device)
            if self.egreedy < torch.rand(1)[0].item():
                q_vals = self.NN(state)
                action = torch.max(q_vals, 0)[1].item()
            else:
                action = self.env.action_space.sample()

        return action

    def optimize(self):
        # q_table[state,action] = (1-alpha) * q_table[state,action] + alpha * (reward + gamma * torch.max(q_table[new_state]))
        if self.memory.__len__() < batch_size:
            return
        state, action, new_state, reward, done = self.memory.sample(batch_size)

        state = torch.Tensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        done = torch.Tensor(done).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor(reward).to(device)


        if double_DQN:
            new_state_indexes = self.NN(new_state).detach()
            maximum_indexes = torch.max(new_state_indexes, 1)[1]
            new_state_values = self.targetNN(new_state).detach()
            maximum = new_state_values.gather(1,maximum_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.targetNN(new_state).detach()
            maximum = torch.max(new_state_values,1)[0]

        target_val = reward + (1-done) * gamma * maximum

        predicted = self.NN(state).gather(1,action.unsqueeze(1)).squeeze(1)

        loss = self.lossFunc(predicted, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        if clip_err:
            for param in self.NN.parameters():
                param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        if self.update_target_counter % update_target_frequency == 0:
            self.targetNN.load_state_dict(self.NN.state_dict())
        self.update_target_counter += 1

    def play(self):
        for episode in range(num_o_ep):
            state = self.env.reset()
            step = 0
            score = 0
            self.statistics.set('episode', episode)


            while True:
                step += 1
                self.statistics.add('frames_total', 1 ,False)

                action = self.get_action(state)
                new_state, reward, done, info = self.env.step(action)
                if (episode % render_frequency == 0):
                    env.render()
                score += reward

                self.memory.push((state, action, new_state, reward, done))

                self.optimize()

                if self.egreedy > egreedy_final:
                    self.egreedy *= decay
                    self.statistics.set('egreedy',self.egreedy)

                state = new_state

                if done:
                    self.statistics.add('steps_total',step, True)
                    self.statistics.add('score_total',score, True)

                    if (episode % weights_saving_frequency == 0):
                        torch.save(self.NN.state_dict(), weights_file)
                    mean_reward_last_100 = self.statistics.mean('score_total',True,100)

                    if mean_reward_last_100 > score_to_achieve:
                        print('solved after %i episodes' % self.statistics.get('solved_after'))
                        if self.statistics.get('solved') == False:
                            self.statistics.set('solved', True)
                            self.statistics.set('solved_after', episode)
                    self.statistics.string_report(report_interval)
                    print('episode finished after %i steps' % step)
                    break

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 23
env = gym.make('LunarLander-v2')
print(env.observation_space)
print(env.action_space)
env.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
agent = Agent(batch_size,env,device)
agent.play()