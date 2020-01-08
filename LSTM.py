import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

SEED = 1

# set seed
np.random.seed(1)

########################
# Activation Functions #
########################

def sigmoid(x: List[int], derivative = False: bool) -> int:
    e_x = np.exp(x) 
    sig = e_x / (e_x + 1)
    if derivative:
        return sig*(1-sig)
    else:
        return sig

def tanh(x: List[int], derivative = False: bool) -> int:
    # formulas via numpy docs
    sinh = 0.5 * (np.exp(x) - np.exp(-x))
    cosh = 0.5 * (np.exp(x) + np.exp(-x))
    tanh = sinh/cosh 

    if derivative:
        return 1 - tanh**2
    else:
        return tanh

def softmax(x: List[int], derivative = False: bool) -> int:
    if derivative:
        pass
    else:
        # to get numerical stability compute in logspace
        return np.exp(x) / sum(np.exp(x))

##########
#  LSTM  #
##########
def class NumpyLSTM(): # Single RNN cell followed by FC layer
    def __init__(hidden_dim: int, input_dim: int):
        # input + hidden vectors are cocatenated
        output_dim = hidden_dim + input_dim

        #### LSTM ####
        # Short-Term Memory (hidden state)
        self.h = np.zeros((hidden_dim, 1))
        # Long-Term Memory (persistent cell state)
        self.c = np.zeros((hidden_dim, 1))

        # Weight matrices for:
        # forget (f), input (i), output (o), and  cell state (c)
        self.W_f = np.random.randn(hidden_dim, output_dim)
        self.W_i = np.random.randn(hidden_dim, output_dim)
        self.W_o = np.random.randn(hidden_dim, output_dim)
        self.W_c = np.random.randn(hidden_dim, output_dim)

        # Biases:
        self.b_f = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((hidden_dim, 1))
        self.b_i = np.zeros((hidden_dim, 1))
        self.b_c = np.zeros((hidden_dim, 1))

        #### Fully Connected Layer ####
        self.W_fc = np.random.randn(input_dim, hidden_dim)
        self.b_fc = np.zeros((input_dim, 1))

    # Cross Entropy
    def loss(pred , targ):
        return -np.mean(np.multiply(np.log(pred), targ))

    # Takes input vector and returns scores for classes
    def forward(x: List[int]) -> List[int]:
        # concat input with hidden state (h = h_{t-1}, x = x_t)
        concat = np.vstack((self.h, x))

        # forget gate activation
        f = sigmoid(np.matmul(self.W_f, concat) + self.b_f)
        # input gate activation
        i = sigmoid(np.matmul(self.W_i, concat) + self.b_i)
        # output gate activation
        o = sigmoid(np.matmul(self.W_o, concat) + self.b_o)
        # update cell state (new candidate vals)
        cand = tanh(np.matmul(self.W_c, concat) + self.b_c)

        # Update state (multiply is elementwise)
        # c_t = f * c_{t-1} + i_t * cand_t
        self.c = np.multiply(f, self.c) +  np.multiply(i, cand)
        self.h = np.multiply(o, tanh(self.c))

        # Fully connected layer 
        fc = np.matmul(self.W_fc, self.h) + self.b_fc # logits
        return softmax(fc) # when using this you want to do np.argmax(output)

    def backward(concat, outputs):
        loss = 0

        for t in range(len(outputs), 0, -1):
            loss += self.loss(pred, targ)

def class PytorchLSTM(nn.Module):
    def __init__(self):
        super(PytorchLSTM, self).__init__()