import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,number_of_classes):
        super(NeuralNetwork, self).__init__()
        self.layer_one = nn.Linear(input_size,hidden_size)
        self.layer_two = nn.Linear(hidden_size,hidden_size)
        self.layer_three = nn.Linear(hidden_size,number_of_classes)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.layer_one(x)
        out = self.relu(out)
        out = self.layer_two(out)

        out = self.relu(out)
        out = self.layer_three(out)

        return out