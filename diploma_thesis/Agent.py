import torch.nn as nn
import torch
import torch.optim as opt

from Neural_network import NeuralNetwork
from Memory import ExperienceReplay
from Statistics import Statistics
# from config import *
import random
import gym

import os

# learn_rate = 0.001
# num_o_ep = 20000
# gamma = 0.99
# egreedy = 0.99
# decay = 0.999
# egreedy_final = 0.01
# score_to_achieve = 200
# report_interval = 10
# solved = False
# hidden_layer = 128
# hidden_layer1 = 64
# hidden_layer2 = 64
# memory_size = 600000
# batch_size = 64
# update_target_frequency = 200
# clip_err = False
# double_DQN = True
# load_weights = False
# weights_file = 'weights_DQN.txt'
# render_frequency = 100
# weights_saving_frequency = 100
# dueling = True

class Agent():
    def __init__(self,config,pipe_in):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(config)
        seed = 23
        self.env = gym.make('LunarLander-v2')
        self.env.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.config = config
        self.pipe_in = pipe_in
        self.num_o_ep = config['num_o_ep']
        self.num_o_input = self.env.observation_space.shape[0]
        self.num_o_output = self.env.action_space.n
        self.NN = NeuralNetwork(self.num_o_input,self.num_o_output,self.config).to(self.device)
        self.targetNN = NeuralNetwork(self.num_o_input,self.num_o_output,self.config).to(self.device)
        self.stats_output_file = config['stats_output_file']
        if self.config['load_weights_enabled']:
            self.NN.load_state_dict(torch.load(self.config['load_weights_filename']))
            self.targetNN.load_state_dict(torch.load(self.config['load_weights_filename']))

        self.lossFunc = nn.MSELoss()
        self.optimizer = opt.Adam(params=self.NN.parameters(), lr=self.config['learn_rate'])
        self.batch_size = self.config['batch_size']
        self.memory = ExperienceReplay(self.config['memory_size'])
        self.statistics = Statistics()
        self.egreedy = self.config['egreedy']
        self.egreedy_final = self.config['egreedy_final']
        self.decay = self.config['decay']
        self.statistics.set('egreedy',self.egreedy)
        self.update_target_counter = 0

        if config['variable_updating_enabled']:
            self.update_frequency = config['update_target_frequency_base']
            self.update_frequency_float = config['update_target_frequency_base']
            self.update_frequency_multiplicator = config['update_target_frequency_multiplicator']
            self.update_frequency_limit = config['update_target_frequency_limit']
        else:
            self.update_frequency = config['update_target_frequency']
        self.statistics.add('update_target_frequency', self.update_frequency, True)


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
        if self.memory.__len__() < self.batch_size:
            return
        state, action, new_state, reward, done = self.memory.sample(self.batch_size)

        state = torch.Tensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)
        new_state = torch.Tensor(new_state).to(self.device)
        reward = torch.Tensor(reward).to(self.device)


        if self.config['double_DQN']:
            new_state_indexes = self.NN(new_state).detach()
            maximum_indexes = torch.max(new_state_indexes, 1)[1]
            new_state_values = self.targetNN(new_state).detach()
            maximum = new_state_values.gather(1,maximum_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.targetNN(new_state).detach()
            maximum = torch.max(new_state_values,1)[0]

        target_val = reward + (1-done) * self.config['gamma'] * maximum

        predicted = self.NN(state).gather(1,action.unsqueeze(1)).squeeze(1)

        loss = self.lossFunc(predicted, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config['clip_err']:
            for param in self.NN.parameters():
                param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        if self.update_target_counter % self.update_frequency == 0:
            self.targetNN.load_state_dict(self.NN.state_dict())
            if self.config['variable_updating_enabled']:
                if self.update_frequency <= self.update_frequency_limit:
                    self.update_frequency_float *= self.update_frequency_multiplicator
                    self.update_frequency = int(self.update_frequency_float)
            self.statistics.add('update_target_frequency', self.update_frequency, True)
        self.update_target_counter += 1

    def play(self):
        for episode in range(self.num_o_ep):
            state = self.env.reset()
            step = 0
            score = 0
            self.statistics.set('episode', episode)


            while True:
                if self.pipe_in.poll() == True:
                    print('im done')
                    self.save_stats()
                    return

                step += 1
                self.statistics.add('frames_total', 1 ,False)

                action = self.get_action(state)
                new_state, reward, done, info = self.env.step(action)
                if self.config['rendering_enabled']:
                    if (episode % self.config['render_frequency'] == 0):
                        self.env.render()
                score += reward

                self.memory.push((state, action, new_state, reward, done))

                self.optimize()

                if self.egreedy > self.egreedy_final:
                    self.egreedy *= self.decay
                    self.statistics.set('egreedy',self.egreedy)

                state = new_state

                if done:
                    self.statistics.add('steps_total',step, True)
                    self.statistics.add('score_total',score, True)
                    if self.config['save_weights_enabled'] == True:
                        if (episode % self.config['weights_saving_frequency'] == 0):
                            torch.save(self.NN.state_dict(), self.config['save_weights_filename'])
                    mean_reward_last_100 = self.statistics.mean('score_total',True,100)

                    if mean_reward_last_100 > self.config['score_to_achieve'] or self.statistics.get('solved') == True:
                        print('solved after %i episodes' % self.statistics.get('solved_after'))
                        if self.statistics.get('solved') == False:
                            self.statistics.set('solved', True)
                            self.statistics.set('solved_after', episode)
                    self.statistics.string_report(self.config['report_interval'])
                    self.send_update()
                    print('episode finished after %i steps' % step)
                    break
        self.save_stats()


    def send_update(self):
        tmp = '{0:.2f},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:d},{6:d},{7:d},{8:d},{9}'.format(self.statistics.get('egreedy'),
              self.statistics.get('score_total')[-1],
              self.statistics.mean('score_total'),
              self.statistics.mean('score_total',True,100),
              self.statistics.mean('score_total', True, self.config['report_interval']),
              self.statistics.get('steps_total')[-1],
              self.statistics.get('episode'),
              self.statistics.get('frames_total'),
              len(self.memory),
              self.statistics.get('solved'))
        self.pipe_in.send(tmp)

    def save_stats(self):
        with open(self.stats_output_file, 'w') as file:
            file.write(self.statistics.to_json())