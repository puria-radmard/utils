import torch
import unittest

from purias_utils.rnn.model.rate import *
from purias_utils.rnn.model.rate_activation import *
from purias_utils.rnn.model.input_nl import *

from purias_utils.rnn.layers.dales import BinaryMaskRecurrent, BinaryMaskForward
from purias_utils.rnn.layers.base import ZeroingWeightLayer, AbsWeightLayer

from purias_utils.noise.ou import OrnsteinUhlenbeckProcess


tau_eta = 0.1
dt = 0.005
eps1=1.0 - (dt/tau_eta)
eps2=(2 * (dt/tau_eta))**0.5

test_rnn_1 = RateRNN(
    rate_activation = make_rectified_power_law_activation_function(0.3, 2),
    input_nonlinearity = linear_input_projection(),
    tau_e=0.1,
    tau_i=0.05,
    min_u_value=200,
    max_u_value=200,
    noise_process=OrnsteinUhlenbeckProcess(torch.eye(100), 1.0, eps1, eps2, 'cuda'),
    num_trials=20,
    device='cuda',
    W_rec=BinaryMaskRecurrent(AbsWeightLayer(torch.randn(100, 100)), exc_indexes=range(80)).cuda(),
    W_input=BinaryMaskForward(torch.randn(100, 9), exc_indexes=range(80), exempt_indices=[0]).cuda(),
    bias=ZeroingWeightLayer(torch.randn(1, 100, 1)).cuda()
)



class TestWeightLayer(unittest.TestCase):

    def test_arg_saving(self):
        """A bit useless if we're saving args anyway!"""
        self.assertTrue(test_rnn_1.rate_activation.k == 0.3, test_rnn_1.rate_activation.gamma == 2.0)

    def test_dynamics(self):
        
        try:
            h = torch.arange(9).unsqueeze(0).repeat(3, 1).float().cuda()
            u0 = torch.randn(3, 100, test_rnn_1.num_trials).float().cuda()
            eta0 = torch.randn_like(u0).cuda()
            test_rnn_1.run_dynamics(u0, h, eta0, 0.005, True, 100, False)
        except Exception as e:
            self.fail(e)




if __name__ == '__main__':

    unittest.main()

