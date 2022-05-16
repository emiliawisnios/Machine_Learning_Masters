import torch
import torch.nn as nn


class DefaultPolicy(nn.Module):
    def __init__(self, hidden_size_pol, hidden_size, db_size, bs_size):
        super(DefaultPolicy, self).__init__()
        self.hidden_size = hidden_size

        self.W_u = nn.Linear(hidden_size, hidden_size_pol, bias=False)
        self.W_bs = nn.Linear(bs_size, hidden_size_pol, bias=False)
        self.W_db = nn.Linear(db_size, hidden_size_pol, bias=False)

    def forward(self, encodings, db_tensor, bs_tensor):
        if isinstance(encodings, tuple):
            hidden = encodings[0]
        else:
            hidden = encodings

        # Network based
        output = self.W_u(hidden[0]) + self.W_db(db_tensor) + self.W_bs(bs_tensor)
        output = torch.tanh(output)

        if isinstance(encodings, tuple):
            # return LSTM tuple if needed
            return (output.unsqueeze(0), encodings[1])
        else:
            return output.unsqueeze(0)


# TODO TASK f)
# Create a policy module that will output the encoded probabilities of 
# 10 actions the the system might choose from.
# Compress information to the hidden_size_pol in the first 2 layers,
# make the policy choose one action through a softmax layer and move from
# the action space back to hidden_size_pol space size.
class SoftmaxPolicy(nn.Module):
    def __init__(self, hidden_size_pol, hidden_size, db_size, bs_size):
        super(SoftmaxPolicy, self).__init__()
        self.hidden_size = None
        self.act_num = None

        self.W_u = None
        self.W_bs = None
        self.W_db = None

        self.first_layer = None
        self.second_layer = None
        self.encoding_layer = None

        # YOUR CODE STARTS HERE:
        self.hidden_size = hidden_size
        self.act_num = 10
        self.W_u = nn.Linear(hidden_size, hidden_size_pol, bias=False)
        self.W_bs = nn.Linear(bs_size, hidden_size_pol, bias=False)
        self.W_db = nn.Linear(db_size, hidden_size_pol, bias=False)
        self.first_layer = nn.Linear(hidden_size_pol, hidden_size_pol, bias=False)
        self.second_layer = nn.Linear(hidden_size_pol, self.act_num, bias=False)
        self.encoding_layer = nn.Linear(self.act_num, hidden_size, bias=False)
        # YOUR CODE ENDS HERE.

    def forward(self, encodings, db_tensor, bs_tensor, act_tensor=None):
        if isinstance(encodings, tuple):
            hidden = encodings[0]
        else:
            hidden = encodings

        # YOUR CODE STARTS HERE:
        act_tensor = self.W_u(hidden[0]) + self.W_db(db_tensor) + self.W_bs(bs_tensor)
        act_tensor = self.first_layer(act_tensor)
        act_tensor = self.second_layer(act_tensor)
        o_t = torch.softmax(act_tensor, dim=1)
        output = self.encoding_layer(o_t)

        # YOUR CODE ENDS HERE.

        if isinstance(encodings, tuple):  # return LSTM tuple
            return (output.unsqueeze(0), encodings[1])
        else:
            return output.unsqueeze(0)

