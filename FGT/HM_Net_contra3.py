import torch
from torch.autograd import Variable, Function
from torch.nn import Module, Parameter
import torch.nn as nn
import HM_LSTM

class HM_Net(Module):
    def __init__(self, a, size_list, dict_size, embed_size,seq_len):
        super(HM_Net, self).__init__()
        self.dict_size = dict_size  # Vocab size
        self.size_list = size_list  # Number of hidden units in each layer
        self.HM_LSTM = HM_LSTM.HM_LSTM(a, embed_size, size_list)
        self.weight = nn.Linear((size_list[0] + size_list[1] + size_list[2])*seq_len, 3)
        self.embed_out1 = nn.Linear(size_list[0]*seq_len, dict_size)
        self.embed_out2 = nn.Linear(size_list[1]*seq_len, dict_size)
        self.embed_out3 = nn.Linear(size_list[2]*seq_len, dict_size)
        self.output_layer = nn.Linear(size_list[0] * seq_len + size_list[1] * seq_len + size_list[2] * seq_len, 2)

    def forward(self, inputs, hidden):
        emb = inputs
        h_1, h_2, h_3, z_1, z_2, z_3, hidden = self.HM_LSTM(emb, hidden)  # batch_size * time_steps * hidden_size
        h = torch.cat((h_1, h_2, h_3), 2)

        g = torch.sigmoid(self.weight(h.view(h.size(0),  h.size(1) * h.size(2))))  # This g is not the LSTM g, but the g combining the different hidden states to predict the outputs
        g_1 = g[:, 0:1]  # batch_size * time_steps, 1
        g_2 = g[:, 1:2]
        g_3 = g[:, 2:3]
        h_e1 = g_1.expand(g_1.size(0), self.dict_size) * self.embed_out1(
            h_1.view(h_1.size(0),  h_1.size(1)*h_1.size(2)))
        h_e2 = g_2.expand(g_2.size(0), self.dict_size) * self.embed_out2(
            h_2.view(h_2.size(0) , h_2.size(1)*h_2.size(2)))
        h_e3 = g_3.expand(g_3.size(0), self.dict_size) * self.embed_out3(
            h_3.view(h_3.size(0) ,  h_3.size(1)*h_3.size(2)))
 
        output = self.output_layer(h.view(h.size(0), h.size(1) * h.size(2)))
        return output
    
    def init_hidden(self, batch_size):
        # Layer 1
        h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
        z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        # Layer 2
        h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
        z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        # Layer 3
        h_t3 = Variable(torch.zeros(self.size_list[2], batch_size).float().cuda(), requires_grad=False)
        c_t3 = Variable(torch.zeros(self.size_list[2], batch_size).float().cuda(), requires_grad=False)
        z_t3 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2, h_t3, c_t3, z_t3)
        return hidden

