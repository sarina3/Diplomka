import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,input, output, config):
        super(NeuralNetwork, self).__init__()
        self.seq = nn.Sequential()
        self.dueling = config['dueling']
        if config['number_o_hidden_layers'] > 0:
            self.num_o_hidden = config['number_o_hidden_layers']
            self.seq.add_module('input', nn.Linear(input,config['hidden_layers'][0]))
            self.add_module('activ_input',nn.ReLU())
            for i in range(1,self.num_o_hidden):
                self.seq.add_module('hidden' + str(i),nn.Linear(config['hidden_layers'][i-1],config['hidden_layers'][i]))
                self.seq.add_module('activ_hidden_' + str(i), nn.ReLU())
            if self.dueling == True:
                self.vVal = nn.Linear(config['hidden_layers'][-1], 1)
                self.adv = nn.Linear(config['hidden_layers'][-1],output)
            else:
                self.output = nn.Linear(config['hidden_layers'][-1],output)

        # ###old####
        # self.input_layer = nn.Linear(input, hidden1)
        # self.hidden = nn.Linear(hidden1,hidden2)
        # self.dueling = dueling_architecture
        # # self.hidden2 = nn.Linear(hidden2,hidden3)
        # if dueling_architecture == False:
        #     self.output = nn.Linear(hidden2, output)
        # else:
        #     self.vVal = nn.Linear(hidden2,1)
        #     self.adv = nn.Linear(hidden2,output)

    def forward(self, state):
        val = self.seq(state)
        if self.dueling == True:
            vVal = self.vVal(val)
            adv = self.adv(val)
            output = vVal + adv - adv.mean()
        else:
            output = self.output(val)
        return output
        ###old###
        # val1 = self.input_layer(state)
        # act_val1 = self.activaton(val1)
        # val2 = self.hidden(act_val1)
        # act_val2 = self.activaton(val2)
        # # val3 = self.hidden2(act_val2)
        # # act_val3 = self.activaton(val3)
        # if self.dueling == False:
        #     output = self.output(act_val2)
        # else:
        #     vVal = self.vVal(act_val2)
        #     adv = self.adv(act_val2)
        #     output = vVal + adv - adv.mean()
        # return output