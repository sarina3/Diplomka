import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,input, hidden1, hidden2,hidden3, output, dueling_architecture):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input, hidden1)
        self.activaton = nn.ReLU()
        self.hidden = nn.Linear(hidden1,hidden2)
        self.dueling = dueling_architecture
        # self.hidden2 = nn.Linear(hidden2,hidden3)
        if dueling_architecture == False:
            self.output = nn.Linear(hidden2, output)
        else:
            self.vVal = nn.Linear(hidden2,1)
            self.adv = nn.Linear(hidden2,output)

    def forward(self, state):
        val1 = self.input_layer(state)
        act_val1 = self.activaton(val1)
        val2 = self.hidden(act_val1)
        act_val2 = self.activaton(val2)
        # val3 = self.hidden2(act_val2)
        # act_val3 = self.activaton(val3)
        if self.dueling == False:
            output = self.output(act_val2)
        else:
            vVal = self.vVal(act_val2)
            adv = self.adv(act_val2)
            output = vVal + adv - adv.mean()
        return output