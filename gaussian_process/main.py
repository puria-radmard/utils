import torch
from torch import nn
from torch import Tensor as _T
from torch.optim.optimizer import Optimizer

import math
from tqdm import tqdm

from purias_utils.gaussian_process.prior import GaussianProcessPrior

class GaussianProcessFit(nn.Module):
    """
        This is all going off the 4F13 notes.
        Noise variance is sigma^2 in the notes
    """

    def __init__(self, prior: GaussianProcessPrior, noise_variance: float):

        super().__init__()
        
        self.prior = prior
        self.posterior_fitted = False
        
        self.noisy_cov_inv: _T = None
        self.fitting_features: _T = None
        self.fitting_outputs: _T = None
        
        self.register_parameter(
            name='log_noise_variance', 
            param=torch.nn.Parameter(torch.tensor(noise_variance).log())
        )
        self.start_sigma_squared = noise_variance

    def reset_state_dict(self):
        self.log_noise_variance.grad = None
        self.log_noise_variance.data = torch.tensor(self.start_sigma_squared).log()
        self.prior.reset_state_dict()

    def nll(self, data_features: _T, data_outputs: _T) -> _T:
        K_and_noisy_inv = self.fit_posterior(data_features, data_outputs, keep = False)
        K = K_and_noisy_inv["K"]
        noisy_cov_inv = K_and_noisy_inv["noisy_cov_inv"]
        noise_matrix = K_and_noisy_inv["noise_matrix"]
        noisy_cov_det = torch.linalg.det(K + noise_matrix)
        n = data_outputs.shape[0]
        return 0.5 * (
            (data_outputs @ noisy_cov_inv @ data_outputs) +
            noisy_cov_det.log() + 
            (n * math.log(2 * torch.pi))
        )

    def fit_prior(self, data_features: _T, data_outputs: _T, optim: Optimizer = None, num_steps: int = 10000, num_repeats: int = 10, show_losses = False, reset=True):
        """
            Fit the parameters of the prior kernel to some data
        """
        if reset:
            self.reset_state_dict()
        print(self.prior.log_dimension_lengths.data.exp().item(), self.prior.log_primary_length.data.exp().item())
        best_nll = float('inf')
        loss_curves = []
        if optim is None:
            optim = torch.optim.Adam(params=self.parameters(), lr = 0.001)
        for rep in range(num_repeats):
            print(f'Fitting GP - repeat {rep+1}')
            new_loss_curve = [self.nll(data_features, data_outputs).item()]
            self.reset_state_dict()
            for t in tqdm(range(num_steps)):
                optim.zero_grad()
                nll = self.nll(data_features, data_outputs)
                new_loss_curve.append(nll.item())
                nll.backward()
                optim.step()
                if show_losses:
                    print(t, '\t', nll.item())
            loss_curves.append(new_loss_curve)
            if new_loss_curve[-1] < best_nll:
                best_nll = new_loss_curve[-1]
                best_state_dict = self.state_dict()
            print(f'Repeat {rep} done. Start NLL = {round(new_loss_curve[0], 3)}. End NLL = {round(new_loss_curve[-1], 3)}.')
        self.load_state_dict(best_state_dict)
        return loss_curves


    def check_fitted(self):
        if not self.posterior_fitted:
            raise Exception('Posterior not fitted to any data yet!')

    def fit_posterior(self, data_features: _T, data_outputs: _T, keep=True):
        """
            Generate the heavy machinery for the posterior, to
                save us having to recalculate for subsequent test points
            
            ^ This is only true if keep=True, otherwise you can get this heavy machinery
                for one time use
        """
        assert tuple(data_outputs.shape) == (data_features.shape[0], )   # Scalar!
        noise_matrix = self.log_noise_variance.exp()  * torch.eye(data_features.shape[0])
        K = self.prior.correlation_matrix(data_features)
        noisy_cov_inv = torch.linalg.inv(K + noise_matrix)
        if keep:
            self.noisy_cov_inv = noisy_cov_inv
            self.fitting_features = data_features
            self.fitting_outputs = data_outputs
            self.posterior_fitted = True
        else:
            return {"K": K, "noisy_cov_inv": noisy_cov_inv, "noise_matrix": noise_matrix}

    @staticmethod
    def make_matrix_psd(matrix):
        ## TODO: make this its own thing!
        eigenvals, eigenvecs = torch.linalg.eig(matrix)
        eigenvals = torch.relu(eigenvals.real) + 1j*eigenvals.imag
        lamb = torch.diag(eigenvals)
        low_rank = eigenvecs @ lamb @ eigenvecs.T
        return low_rank.real

    def predictive_distribution(self, data):
        """
            Returns the posterior statistics for the features provided
            i.e. m_post(x) and k_post(x, x') evaluated at the data features
        """
        self.check_fitted()

        kernel_vector = self.prior.correlation_matrix(self.fitting_features, data)
        m_post = kernel_vector @ self.noisy_cov_inv @ self.fitting_outputs

        prior_cov = self.prior.correlation_matrix(data, data)
        K_post = prior_cov - (kernel_vector @ self.noisy_cov_inv @ kernel_vector.T)
        # torch.linalg.cholesky(prior_cov)

        return {'m_post': m_post, 'K_post': K_post}

    def sample_from_posterior(self, m_post: _T, K_post: _T, make_K_post_psd= False):
        if make_K_post_psd:
            K_post = self.make_matrix_psd(K_post)
        seed = torch.randn(m_post.shape[0], device=m_post.device)
        cov_chol = torch.linalg.cholesky(K_post)# + 1e-1 * torch.eye(K_post.shape[0]))
        return (cov_chol @ seed) + m_post

