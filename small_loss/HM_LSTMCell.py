import torch, math
from torch.autograd import Variable, Function
from torch.nn import Module, Parameter

# A modification of the simoid function, as described in the article. a defines the slope hard_sigm(x) = max(0,min(1,(ax+1)/2))
def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

class Bound(Function):
    @staticmethod
    def forward(ctx, x):
        # forward : x -> output
        ctx.save_for_backward(x)
        output = x > 0.5
        return output.float()

    @staticmethod
    def backward(ctx, output_grad):
        # backward: output_grad -> x_grad
        x, = ctx.saved_tensors
        x_grad = None

        if ctx.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad

# 使用方法
bound = Bound.apply

class HM_LSTMCell(Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        # HM_LSTMCell(self.input_size, self.size_list[0], self.size_list[1], self.a, False)
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size 
        self.top_size = top_size
        self.a = a #se slope annealing trick i training
        self.last_layer = last_layer


        #Initialize weight matrices for transition of hidden states between HM_LSTM cells
        '''
        U_recur means the state transition parameters from layer l (current layer) to layer l
        U_topdown means the state transition parameters from layer l+1 (top layer) to layer l
        W_bottomup means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.W_bottomup = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.bottom_size))
        self.U_recur = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))

        if not self.last_layer:
            self.U_topdown = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1, self.top_size))

        self.bias = Parameter(torch.cuda.FloatTensor(4 * self.hidden_size + 1))

        #Perform weight initialization of these 4 (or 3) parameters with function defined below.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottomup, h_recur, h_topdown, z, z_bottom): 
        # h_bottom.size = bottom_size * batch_size
        """
        c:                  cell state in previous cell (l,t-1)
        h_bottomup:         hidden states in layer below (l-1,t)
        h_recur:            hidden states in previous timestep (l,t-1)
        h_topdown:          hidden states in layer above and previous timestep (l+1,t-1)
        z:                  boundary state in previous time step (l,t-1)
        z_bottom:           boundary state in layer below (l-1,t)

        """
        #Calculate s-matrices to calculate new cell state (COPY,UPDATE or FLUSH)
        s_recur = torch.mm(self.U_recur, h_recur)
        h_bottomup = torch.tensor(h_bottomup, dtype=torch.float32)
        h_bottomup = h_bottomup.cuda()
        s_bottomup_init = torch.mm(self.W_bottomup, h_bottomup)
        s_bottomup = z_bottom.expand_as(s_bottomup_init) * s_bottomup_init

        #If not last layer, calculate s_topdown, else set s_topdown to 0
        if not self.last_layer:
            s_topdown_init = torch.mm(self.U_topdown, h_topdown) 
            s_topdown = z.expand_as(s_topdown_init) * s_topdown_init
        else:
            s_topdown = Variable(torch.zeros(s_recur.size()).cuda(), requires_grad=False).cuda()
        #Extract individual variables (matrix/vector) from the large S matrix (f_s: f_slice) 
        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        f = torch.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = torch.sigmoid(f_s[self.hidden_size:self.hidden_size*2, :])
        o = torch.sigmoid(f_s[self.hidden_size*2:self.hidden_size*3, :])
        g = torch.tanh(f_s[self.hidden_size*3:self.hidden_size*4, :])
        z_hat = hard_sigm(self.a, f_s[self.hidden_size*4:self.hidden_size*4+1, :])
        # Make vector of ones and resize the boundary states (z-values) to be the same size as the cell parameters (f, i, o and g)
        one = Variable(torch.ones(f.size()).cuda(), requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)
        #Calculate cell state (one line implementation of out commented if/else below)
        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g) #burde ikke beregnes for ethvert tilfælde? 
        h_new = z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h_recur + (one - z) * z_bottom * o * torch.tanh(c_new)
        bound = Bound.apply
        z_new = bound(z_hat)
        return h_new, c_new, z_new