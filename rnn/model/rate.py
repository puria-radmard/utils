import torch
from torch import nn
from torch import Tensor as _T
from torch.nn import functional as F
from tqdm import tqdm

from typing import Type

from purias_utils.ssm.process.base import ProcessBase
from purias_utils.rnn.layers.base import WeightLayer

from purias_utils.util.api import return_as_obj



class RateRNN(nn.Module):

    """
    NB: all parameters must be of type WeightLayer, even if it doesn't make sense!
    
    The names of the parameters will be refered to in the rate_activation and input_nonlinearity,
        so should be named accordingly

    Even if num_trials = 1, an extra dimension is created for all activities/rates

    Assumes only one parameter, which is W_rec which is a Dales BinaryMaskRecurrent
    """

    def __init__(
        self,
        rate_activation: callable,
        tau_e: float,
        tau_i: float,
        noise_process: ProcessBase,
        device = 'cuda',
        min_e_u: float = None,
        max_e_u: float = None,
        min_i_u: float = None,
        max_i_u: float = None,
        **parameters: Type[WeightLayer],
    ):

        super(RateRNN, self).__init__()

        for k, v in parameters.items():
            assert isinstance(v, WeightLayer)
            self.add_module(k, v.to(device))
        self.parameter_names = list(parameters.keys())

        self.device = device
        self.n_h = self.W_rec.num_neurons
        self.noise_process = noise_process
        self.rate_activation = rate_activation

        self.tau_e = tau_e
        self.tau_i = tau_i

        # For clipping use, TODO: properorty setter for any of these
        u_max = torch.ones(1, 1, self.n_h)
        u_min = torch.ones(1, 1, self.n_h)
        self.min_e_u = min_e_u or -float('inf')
        self.max_e_u = max_e_u or float('inf')
        self.min_i_u = min_i_u or -float('inf')
        self.max_i_u = max_i_u or float('inf')
        u_max[...,list(self.W_rec.exc_indexes)] = self.max_e_u
        u_min[...,list(self.W_rec.exc_indexes)] = self.min_e_u
        u_max[...,list(self.W_rec.inh_indexes)] = self.max_i_u
        u_min[...,list(self.W_rec.inh_indexes)] = self.min_i_u
        self.u_max = u_max.to(device)
        self.u_min = u_min.to(device)

    def to(self, device, *args, **kwargs):
        self.device = device
        self.noise_process.to(self.device)
        self.u_max = self.u_max.to(device)
        self.u_min = self.u_min.to(device)
        return super().to(device, *args, **kwargs)
        
    def cpu(self) -> _T:
        self.device = 'cpu'
        self.noise_process.to(self.device)
        self.u_max = self.u_max.cpu()
        self.u_min = self.u_min.cpu()
        for mod in list(self.modules())[1:]:
            mod.cpu()
        return super().cpu()

    def cuda(self, device = None) -> _T:
        self.device = device or 'cuda'
        self.noise_process.to(self.device)
        self.u_max = self.u_max.cuda()  # Might need to be specified?
        self.u_min = self.u_min.cuda()
        return super().cuda(device)

    def full_mitosis(self, chosen_i):
        raise NotImplementedError("Don't be lazy with full_mitosis!")

    @property
    def tau_inv_matrix(self):
        tau_inv_vector = 1 / torch.tensor(
            [self.tau_e if self.W_rec.is_exc(i) else self.tau_i for i in range(self.n_h)]
        )
        return torch.diag(tau_inv_vector).to(self.device)

    def clip(self, u: _T):
        u_max = self.u_max.repeat(u.shape[0], u.shape[1], 1)
        u_min = self.u_min.repeat(u.shape[0], u.shape[1], 1)
        u = torch.minimum(u, u_max)
        u = torch.maximum(u, u_min)
        return u

    @return_as_obj
    def check_shapes(self, u0: _T, fs: _T, eta0: _T):
        B, _tr, T, n_f = fs.shape                                    # Complete rehaul!!
        assert n_f == self.n_h
        assert list(u0.shape) == list(eta0.shape) == [B, _tr, self.n_h]
        return {'B': B, '_tr': _tr, 'T': T}

    @return_as_obj
    def dynamics_step(self, u: _T, f: _T, eta: _T, dt: float, tau_inv):
        """
        all tensors [batch, trials, nh]
        """
        r = self.rate_activation(u)
        du_dt = (-u + f + self.W_rec(r) + self.bias.masked_weight + eta) @ tau_inv
        return {"u": u + (du_dt * dt), "du_dt": du_dt}

    @return_as_obj
    def run_dynamics(self, u0: _T, fs: _T, eta0: _T, dt: float):

        dims = self.check_shapes(u0, fs, eta0)
        
        u_history, du_dt_history = [], []
        u, eta = u0, eta0
        tau_inv = self.tau_inv_matrix
        self.noise_process.reinitialise(eta0)
        for t in range(dims.T):
            dyn = self.dynamics_step(u = u, f = fs[:,:,t], eta = eta, dt = dt, tau_inv = tau_inv)
            u, eta = self.clip(dyn.u), self.noise_process()
            u_history.append(u)
            du_dt_history.append(dyn.du_dt)
        
        if dims.T > 0:
            u_history = torch.stack(u_history, 2)
            du_dt_history = torch.stack(du_dt_history, 2)
        else:
            u_history = u0.clone().unsqueeze(2)
            du_dt_history = None

        return {
            "u_history": u_history,         # [batch, trials, time, nh]
            "eta": eta,                     # [batch, trials, nh]
            "du_dt_history": du_dt_history  # [batch, trials, time, nh]
        }


class RateRNNOuterActivation(RateRNN):

    @return_as_obj
    def dynamics_step(self, u: _T, f: _T, eta: _T, dt: float, tau_inv):
        """
            Names of variables end up being a bit funky but ah well
            Comparing to Yang paper:
                u here is r there
                r here is f(...) there
                f here is W^{in}u there

            So the output of the network is still u
        """
        r = self.rate_activation(f + self.W_rec(u) + self.bias.masked_weight + eta)
        du_dt = tau_inv @ (-u + r)
        return {"u": u + (du_dt * dt), "du_dt": du_dt}


class OutputFeedbackRateRNN(RateRNN):

    def __init__(self, rate_activation: callable, output_feedback_nonlinearity: callable, tau_e: float, tau_i: float, noise_process: ProcessBase, device='cuda', min_e_u: float = None, max_e_u: float = None, min_i_u: float = None, max_i_u: float = None, **parameters: Type[WeightLayer]):
        super().__init__(rate_activation, tau_e, tau_i, noise_process, device, min_e_u, max_e_u, min_i_u, max_i_u, **parameters)
        self.output_feedback_nonlinearity = output_feedback_nonlinearity

    @return_as_obj
    def dynamics_step(self, u: _T, r: _T, z: _T, f: _T, eta: _T, dt: float, tau_inv):
        zfb = self.output_feedback_nonlinearity(self, z)
        du_dt = tau_inv @ (-u + f + self.W_rec(r) + zfb + self.bias.masked_weight + eta)
        return {"u": u + (du_dt * dt), "du_dt": du_dt}

    @return_as_obj
    def run_dynamics(self, u0: _T, h: _T, eta0: _T, dt: float):
        raise NotImplementedError('deal with new format')
        self.train(not burn_in)
        assert len(h.shape) == 2, "Can only accept 2 axis static inputs to SSN"
        u_history = []
        output_history = []
        du_dt_history = []
        tau_inv = self.tau_inv_matrix
        u = u0
        eta = eta0
        self.noise_process.reinitialise(eta0)
        for _ in tqdm(range(num_steps), disable = not enable_tqdm):
            r = self.rate_activation(u)
            z = self.W_output(r) + self.bias_output.masked_weight
            u, du_dt = self.dynamics_step(u = u, r = r, z = z, f = f, eta = eta, dt = dt, tau_inv = tau_inv)
            u = self.clip(u)
            eta = self.noise_process()
            u_history.append(u)
            output_history.append(z)
            if return_differential:
                du_dt_history.append(du_dt)
        return {"u": u + (du_dt * dt), "du_dt": du_dt}

