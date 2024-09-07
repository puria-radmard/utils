import torch
from torch import nn
from torch import Tensor as _T

from purias_utils.rnn.model.rate import RateRNN, return_as_obj
from purias_utils.maths.matmul import multiply_2d_state_safely


class VisionRNNOutputActivation(RateRNN):

    """
    Torch translation of Wayne's tensorflow code.
    Here for activation == 'relu_output'

    Assumes a W_rec and a W_rec_trans, 
        both Dales BinaryMaskRecurrents

    In Wayne's code:
        - Rate activation is always standard ReLU
        - u comes in as shape [batch, P, C, 1] 
            but we add trials as a 2nd dimension
        - f (the input) is just the 'raw' image projected and scaled scaled, so of course is the
            same shape when it goes in. We also unsqueeze(-1)
    
    Shapes in the du_dt line:
        tau_inv = [image_y, image_x]
        u, all_rec, eta = [batch, trials, image_y, image_x, channels]
        bias = [1, trials, image_y, image_x, channels]
        f = [batch, trials, image_y, image_x, channels]
    """

    def __init__(self, *args, **kwargs):
        super(VisionRNNOutputActivation, self).__init__(*args, **kwargs)
        print('WARNING: VisionRNNOutputActivation.clip requires reworking')

    @return_as_obj
    def check_shapes(self, u0: _T, fs: _T, eta0: _T):
        B, _tr, T, n_P, n_C, C = fs.shape                                    # Complete rehaul!!
        assert list(u0.shape) == list(eta0.shape) == [B, _tr, n_P, n_C, C]
        return {'B': B, '_tr': _tr, 'T': T}

    def clip(self, u: _T):
        return u

    @return_as_obj
    def dynamics_step(self, u: _T, f: _T, eta: _T, dt: float, tau_inv: _T):
        r = self.rate_activation(u)
        r_trans = self.rate_activation(u.transpose(2, 3))
        reg_rec = multiply_2d_state_safely(self.W_rec, r)
        trans_rec = multiply_2d_state_safely(self.W_rec_trans, r_trans)
        all_rec = reg_rec + trans_rec.transpose(2, 3)
        unscaled_du_dt = (-u + all_rec + self.bias.masked_weight + eta + f)
        du_dt = multiply_2d_state_safely(lambda x: tau_inv @ x, unscaled_du_dt.transpose(2, 3))
        du_dt = du_dt.transpose(2, 3)
        return {"u": u + (du_dt * dt), "du_dt": du_dt}


class VisionRNNInputActivation(VisionRNNOutputActivation):

    """
    Torch translation of Wayne's tensorflow code.
    Here for activation == 'relu_input'

    Assumes a W_rec and a W_rec_trans,
        both Dales BinaryMaskRecurrents
    """

    def __init__(self, *args, **kwargs):
        raise Exception("# Both should work for Wayne's weights, but relu_input doesn't!")
        super(VisionRNNInputActivation, self).__init__(*args, **kwargs)

    @return_as_obj
    def dynamics_step(self, u: _T, f: _T, eta: _T, dt: float, tau_inv):
        reg_rec = multiply_2d_state_safely(self.W_rec, u)
        trans_rec = multiply_2d_state_safely(self.W_rec_trans, u.transpose(1, 2))
        all_rec = self.rate_activation(reg_rec + trans_rec.transpose(1, 2))
        unscaled_du_dt = (-u + all_rec + self.bias.masked_weight + eta + f)
        du_dt = multiply_2d_state_safely(lambda x: tau_inv @ x, unscaled_du_dt.transpose(1, 2))
        du_dt = du_dt.transpose(1, 2)
        return {"u": u + (du_dt * dt), "du_dt": du_dt}





class resnetrnn8(nn.Module):

    alpha = 0.2

    def __init__(self, dataset):

        super().__init__()

        if dataset == 'imagenet':
            N_class = 1000

        if dataset == 'cifar10':
            N_class = 10

        self.relu = nn.ReLU(inplace=True)

        self.inp_conv = nn.utils.weight_norm(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3, bias=False))
        self.inp_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.inp_skip = nn.utils.weight_norm(nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3, bias=False))

        self.channels = [64,64,128,128,256,256,512,512]
        self.sizes = [56,56,28,28,14,14,7,7]
        self.strides = [1,2,1,2,1,2,1]
        self.skipchannels = [64,128,256,512]

        area_conv = []
        area_bias = []
        area_norm = []
        for i in range(8):
            area_conv.append(nn.utils.weight_norm(nn.Conv2d(self.channels[i],self.channels[i],kernel_size=3,stride=1,padding=1,bias=False)))
            area_bias.append(nn.Parameter(torch.randn(self.channels[i],self.sizes[i],self.sizes[i]), requires_grad=True))
            area_norm.append(nn.BatchNorm2d(num_features=self.channels[i]))
        self.area_conv = nn.ParameterList(area_conv)
        self.area_bias = nn.ParameterList(area_bias)
        self.area_norm = nn.ParameterList(area_norm)

        area_area = []
        for i in range(7):
            area_area.append(nn.utils.weight_norm(nn.Conv2d(self.channels[i],self.channels[i+1],kernel_size=3,stride=self.strides[i],padding=1,bias=False)))
        self.area_area = nn.ParameterList(area_area)

        skip_area = []
        for i in range(3):
            skip_area.append(nn.utils.weight_norm(nn.Conv2d(self.skipchannels[i],self.skipchannels[i+1],kernel_size=3,stride=2,padding=1,bias=False)))
        self.skip_area = nn.ParameterList(skip_area)

        self.sensory_size = 512

        self.out_avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False)
        self.out_flatten = nn.Flatten()
        self.out_fc = nn.utils.weight_norm(nn.Linear(self.sensory_size,N_class,bias=False))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x,a0,img,alpha=None):

        a = a0.clone()
        alpha = alpha or self.alpha

        inp0 = self.inp_avgpool(self.inp_conv(img))
        inp1 = self.inp_avgpool(self.inp_skip(img))
        
        x[7] = (1-alpha)*x[7] + alpha*self.relu(self.area_conv[7](x[7]) + self.area_bias[7] + self.area_area[6](x[6]       ) + self.skip_area[2](x[5]) )
        x[6] = (1-alpha)*x[6] + alpha*self.relu(self.area_conv[6](x[6]) + self.area_bias[6] + self.area_area[5](x[5] + x[4])                           )
        x[5] = (1-alpha)*x[5] + alpha*self.relu(self.area_conv[5](x[5]) + self.area_bias[5] + self.area_area[4](x[4]       ) + self.skip_area[1](x[3]) )
        x[4] = (1-alpha)*x[4] + alpha*self.relu(self.area_conv[4](x[4]) + self.area_bias[4] + self.area_area[3](x[3] + x[2])                           )
        x[3] = (1-alpha)*x[3] + alpha*self.relu(self.area_conv[3](x[3]) + self.area_bias[3] + self.area_area[2](x[2]       ) + self.skip_area[0](x[1]) )
        x[2] = (1-alpha)*x[2] + alpha*self.relu(self.area_conv[2](x[2]) + self.area_bias[2] + self.area_area[1](x[1] + x[0])                           )
        x[1] = (1-alpha)*x[1] + alpha*self.relu(self.area_conv[1](x[1]) + self.area_bias[1] + self.area_area[0](x[0]       ) + inp1                    )
        x[0] = (1-alpha)*x[0] + alpha*self.relu(self.area_conv[0](x[0]) + self.area_bias[0] + inp0                                                     )  
        
        for i in range(8):
            a += x[i].mean((1,2,3))/8
        
        out = self.out_avgpool(x[7])
        out = self.out_flatten(out)
        # out = self.dropout(out)
        # out = self.out_fc(out)

        return out, x, a




