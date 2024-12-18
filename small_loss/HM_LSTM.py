from HM_LSTMCell import HM_LSTMCell
import torch
from torch.autograd import Variable
from torch.nn import Module
class HM_LSTM(Module): #Istedet for at lave egen LSTM så brug den fra main.py
    def __init__(self, a, input_size, size_list):
        super(HM_LSTM, self).__init__()
        self.a = a
        self.input_size = input_size 
        self.size_list = size_list 
        
        """
        input_size:        Input size to network
        size_list[0]:      hidden size of layer 1 aka. input size after embedding
        size_list[1]:      hidden size of layer 2
        size_list[2]:      hidden size of layer 3
        """
        self.cell_1 = HM_LSTMCell(self.input_size, self.size_list[0], self.size_list[1], self.a, False) #bottom_size, hidden_size, top_size, a, last_layer
        self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], self.size_list[2], self.a, False)
        self.cell_3 = HM_LSTMCell(self.size_list[1], self.size_list[2], None, self.a, True)

    def forward(self, inputs, hidden): #hidden state supplied, if the model is going to freestyle or make predictions that dont start from 0
        
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden == None:
            #Layer 1
            h_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
            c_t1 = Variable(torch.zeros(self.size_list[0], batch_size).float().cuda(), requires_grad=False)
            z_t1 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
            #Layer 2
            h_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
            c_t2 = Variable(torch.zeros(self.size_list[1], batch_size).float().cuda(), requires_grad=False)
            z_t2 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
            #Layer 3
            h_t3 = Variable(torch.zeros(self.size_list[2], batch_size).float().cuda(), requires_grad=False)
            c_t3 = Variable(torch.zeros(self.size_list[2], batch_size).float().cuda(), requires_grad=False)
            z_t3 = Variable(torch.zeros(1, batch_size).float().cuda(), requires_grad=False)
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2, h_t3, c_t3, z_t3) = hidden

        #Make vector of ones with the same size as batch, so that input is always passed via s_bottomup (see HMLSTMCell)
        z_one = Variable(torch.ones(1, batch_size).float().cuda(), requires_grad=False)

        """
        h_t1:     hidden states of layer 1 at previous timestep (updates to current in each iteration)
        h_t2:     hidden states of layer 2 at previous timestep (updates to current timestep in each iteration)
        h_t3:     hidden states of layer 3 --||--
        z_1:      boundary state of layer 1 --||--
        z_2:      boundary state of layer 2 --||--
        z_3:      boundary state of layer 3 --||--
        """
        h_1, h_2, h_3, z_1, z_2, z_3 = [], [], [], [], [], []

        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottomup=inputs[:,t, :].t(), h_recur=h_t1, h_topdown=h_t2, z=z_t1, z_bottom=z_one) #t1 og t2 is layer1 and layer2 and not time1 and time2
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottomup=h_t1, h_recur=h_t2, h_topdown=h_t3, z=z_t2, z_bottom=z_t1)
            h_t3, c_t3, z_t3 = self.cell_3(c=c_t3, h_bottomup=h_t2, h_recur=h_t3, h_topdown=None, z=z_t3, z_bottom=z_t2)

            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            h_3 += [h_t3.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]
            z_3 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2, h_t3, c_t3, z_t3)
        return torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(h_3, dim=1), torch.stack(z_1, dim=1), torch.stack(z_2, dim=1), torch.stack(z_3, dim=1), hidden
