import torch
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from tqdm import tqdm

from typing import Type

from purias_utils.noise.base import NoiseProcess
from purias_utils.rnn.layers.base import WeightLayer



class RateRNN(nn.Module):

    """
    NB: all parameters must be of type WeightLayer, even if it doesn't make sense!
    
    The names of the parameters will be refered to in the rate_activation and input_nonlinearity,
        so should be named accordingly

    Even if num_trials = 1, an extra dimension is created for all activities/rates
    """

    def __init__(
        self,
        rate_activation: callable,
        input_nonlinearity: callable,
        tau_e: float,
        tau_i: float,
        min_u_value: float,
        max_u_value: float,
        noise_process: NoiseProcess,
        num_trials: int,
        device = 'cuda',
        **parameters: Type[WeightLayer],
    ):

        super(RateRNN, self).__init__()

        self.device = device

        self.rate_activation = rate_activation
        self.input_nonlinearity = input_nonlinearity

        self.tau_e = tau_e
        self.tau_i = tau_i
        self.min_u_value = min_u_value
        self.max_u_value = max_u_value
        self.noise_process = noise_process
        self.num_trials = num_trials

        for k, v in parameters.items():
            assert isinstance(v, WeightLayer)
            self.add_module(k, v)

    def full_mitosis(self, chosen_i):
        raise NotImplementedError("Don't be lazy with full_mitosis!")

    @property
    def tau_inv_matrix(self):
        tau_inv_vector = 1 / torch.tensor(
            [self.tau_e if self.W_rec.is_exc(i) else self.tau_i for i in range(self.W_rec.num_neurons)]
        )
        return torch.diag(tau_inv_vector).to(self.device)

    def clip(self, u: T):
        if self.min_u_value == self.max_u_value == None:
            return u
        return u.clip(min = self.min_u_value, max = self.max_u_value)

    def _dynamics_step(self, u: T, f: T, eta: T, dt: float, tau_inv):
        r = self.rate_activation(u)
        du_dt = tau_inv @ (-u + f + self.W_rec(r) + self.bias.masked_weight + eta)
        import pdb; pdb.set_trace()
        return u + (du_dt * dt), du_dt

    def run_dynamics(self, u0: T, h: T, eta0: T, dt: float, burn_in: bool, num_steps: int, enable_tqdm=False, return_differential=False):
        self.train(not burn_in)
        assert len(h.shape) == 2, "Can only accept 2 axis inputs to SSN"
        f = self.input_nonlinearity(self, h.unsqueeze(-1)).repeat(1, 1, self.num_trials)
        u_history = []
        du_dt_history = []
        tau_inv = self.tau_inv_matrix
        u = u0
        eta = eta0
        self.noise_process.reinitialise(eta0)
        for _ in tqdm(range(num_steps), disable = not enable_tqdm):
            u, du_dt = self._dynamics_step(u = u, f = f, eta = eta, dt = dt, tau_inv = tau_inv)
            u = self.clip(u)
            eta = self.noise_process()
            u_history.append(u)
            if return_differential:
                du_dt_history.append(du_dt)
        return (u_history, eta, du_dt_history) if return_differential else (u_history, eta)


class RateRNNOuterActivation(RateRNN):

    def _dynamics_step(self, u: T, f: T, eta: T, dt: float, tau_inv):
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
        return u + (du_dt * dt), du_dt


class OutputFeedbackRateRNN(RateRNN):

    def __init__(
            self, rate_activation: callable, input_nonlinearity: callable, output_feedback_nonlinearity: callable, 
            tau_e: float, tau_i: float, min_u_value: float, max_u_value: float, noise_process: NoiseProcess, 
            num_trials: int, device='cuda', **parameters: Type[WeightLayer]
        ):
        super(OutputFeedbackRateRNN, self).__init__(rate_activation, input_nonlinearity, tau_e, tau_i, min_u_value, max_u_value, noise_process, num_trials, device, **parameters)
        self.output_feedback_nonlinearity = output_feedback_nonlinearity

    def _dynamics_step(self, u: T, r: T, z: T, f: T, eta: T, dt: float, tau_inv):
        zfb = self.output_feedback_nonlinearity(self, z)
        du_dt = tau_inv @ (-u + f + self.W_rec(r) + zfb + self.bias.masked_weight + eta)
        return u + (du_dt * dt), du_dt

    def run_dynamics(self, u0: T, h: T, eta0: T, dt: float, burn_in: bool, num_steps: int, enable_tqdm=False, return_differential=False):
        self.train(not burn_in)
        assert len(h.shape) == 2, "Can only accept 2 axis inputs to SSN"
        f = self.input_nonlinearity(self, h.unsqueeze(-1)).repeat(1, 1, self.num_trials)
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
            u, du_dt = self._dynamics_step(u = u, r = r, z = z, f = f, eta = eta, dt = dt, tau_inv = tau_inv)
            u = self.clip(u)
            eta = self.noise_process()
            u_history.append(u)
            output_history.append(z)
            if return_differential:
                du_dt_history.append(du_dt)
        return (u_history, eta, du_dt_history) if return_differential else (u_history, eta)
