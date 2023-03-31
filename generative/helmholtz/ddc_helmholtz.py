import torch
from torch import nn
from torch import Tensor as T
from torch.nn import ParameterList, ModuleList
from torch.nn.functional import sigmoid, mse_loss

from purias_utils.exponential_family_models.zdeep import DeepExponentialFamilyGenerativeModel

from typing import List


class DDCHMLayer(nn.Module):
    """
        This class contains everything pertaining to the lth layer in the paper.
        Namely:
            Nonlinear encodings T, equation 4
            Recongition layer r, equation 8
            Approximations, equations 31-34

        latent_dim: dimension of z_l (or x if l = 0)
        encoding_dim: dimension of T(z_l) (called K, see equation 4)
        lower_encoding_dim: dimension of T(z_{l-1}), i.e. one closer to the data
        
        alpha_output_dim is size of the raw parameter for the corresponding layer (equation 33)
        beta_output_dim is size of the raw parameter for the layer closer to the data (equation 32)
    """
    def __init__(self, latent_dim: int, encoding_dim: int, lower_encoding_dim: int, alpha_output_dim: int, beta_output_dim: int) -> None:
        super().__init__()

        # T(z) - see equation 4
        self.latent_encoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=encoding_dim),
            nn.Sigmoid()
        )
        for param in self.latent_encoder.parameters():
            param.requires_grad = False

        # r_l(x) - see equation 8. NB: recognition done from sleep data 'directly',
        #   does not use sleep latents
        # TODO: DDCHMDataLayer needed for LHS of equation 8 here
        self.recognition_layer = nn.Linear(
            in_features=lower_encoding_dim, out_features=encoding_dim, bias=False
        )

        # Expected gradient approximators, e.g. equation 30
        self.alphas = nn.Linear(
            in_features=encoding_dim, out_features=alpha_output_dim, bias=False
        )

        self.betas = nn.Linear(
            in_features=encoding_dim, out_features=beta_output_dim, bias=False
        )

    def T_enc(self, z: T):
        "Encoding functions remain random"
        with torch.no_grad():
            return self.latent_encoder(z)

    def approximate_with_alphas(self, encoding: T = None, z: T = None):
        "Equation 30/33. Either directly from encoding (T), or with z"
        if encoding is None:
            encoding = self.T_enc(z)
        else:
            assert z is None, "Cannot provide both encoding (T(z)) and latent (z)!"
        return self.alphas(encoding)

    def approximate_with_betas(self, encoding: T = None, z: T = None):
        "Equation 32. Either directly from encoding (T), or with z"
        if encoding is None:
            encoding = self.T_enc(z)
        else:
            assert z is None, "Cannot provide both encoding (T(z)) and latent (z)!"
        return self.betas(encoding)


class DDCHMDataLayer(DDCHMLayer):
    "alphas now multiply not own recognition, but those of next layer up - see equation 36"

    def __init__(self, data_dim: int, encoding_dim: int, sufficient_statistic_dim: int, higher_encoding_dim: int) -> None:

        # bypass DDCHMLayer.__init__()
        super(DDCHMLayer, self).__init__()
        
        # This is actually h(x, W) as in equation 8
        linear = nn.Linear(
            in_features=data_dim, out_features=encoding_dim, bias=False
        )
        self.recognition_layer = nn.Sequential(linear, nn.ReLU())

        alpha_output_dim = sufficient_statistic_dim * sufficient_statistic_dim
        self.sufficient_statistic_dim = sufficient_statistic_dim

        # Special case, e.g. equation 36
        self.alphas = nn.Linear(
            in_features=higher_encoding_dim, out_features=alpha_output_dim, bias=False
        )

    def approximate_with_alphas(self, encoding: T = None, z: T = None):
        """
            Special case - see equation 36.
            TODO: make an alpha0 for every layer? Not used but grammatically correct!
            Equation 30/33. Either directly from encoding (T), or with z
        """
        if encoding is None:
            encoding = self.T_enc(z)
        else:
            assert z is None, "Cannot provide both encoding (T(z)) and latent (z)!"
        return self.alphas(encoding).reshape(-1, self.sufficient_statistic_dim, self.sufficient_statistic_dim)




class DDCHelmholtzMachine(nn.Module):
    """
    Provide with:
        generative_model, a DeepExponentialFamilyGenerativeModel which we are trying to learn
        
        A list of DDCHMLayers. Because z_L is first in DeepExponentialFamilyGenerativeModel,
            unfortunately the DDCHMLayers will also have to be in this reverse order.

            The *last* of these has to be a DDCHMDataLayer
    """
    def __init__(
        self, 
        generative_model: DeepExponentialFamilyGenerativeModel,
        layers: List[DDCHMLayer]
    ):
        super().__init__()
        assert isinstance(layers[-1], DDCHMDataLayer)

        self.layers: List[DDCHMLayer] = ModuleList(layers)      # Length L + 1
        self.generative_model = generative_model
    
    def dream_all_latents(self, batch_size: int):
        "Starting with z_L, generate with the current model (theta), with the last item being dreamt data"
        return self.generative_model.full_generate(batch_size=batch_size)

    def get_all_encodings(self, latents: List[T]):
        "Get T for each layer. Will have no grad. Order is backwards (L -> 1)"
        return [layer.T_enc(latents[i]) for i, layer in enumerate(self.layers[:-1])]

    def full_recognition(self, data: T):
        """
            Loop over layers l and recognise from l-1. Start with data project, which is not passed on
            As noted in docstring, this requires order reversal. 
                We pass this on in the reverse order (L, L-1, ..., 1),
                without the data projection (l=0), as that is not used (h in equation 8)
        """
        data_projection = self.layers[-1].recognition_layer(data)
        output = [data_projection]
        for l in range(2, len(self.layers) + 1):
            next_layer = self.layers[-l]
            latent = next_layer.recognition_layer(output[0])
            output.insert(0, latent)
        return output[:-1]
    
    def sleep_phase_losses(self, batch_size: int):
        """
        Part 1:
            Equation 7. 
            Training recognition model to approximate the (rich) random encodings of the dreamt latents,
                so that it converges on the expected value under the variational distribution q

        Part 2:
            Training function approximation on the gradient flows of the generative model, evaluated on the
                dreamt data, so that it also converges to expect values
        """

        # Part 1:
        dreamt_latents = self.dream_all_latents(batch_size=batch_size)  # Dreamt data is last entry
        dreamt_data_encodings = self.get_all_encodings(dreamt_latents)  # Last entry (dreamt data) is ignored
        recognition_outputs = self.full_recognition(dreamt_latents[-1])

        recognition_losses = []
        for d, r in zip(recognition_outputs, dreamt_data_encodings):
            assert d.shape == r.shape
            recognition_losses.append(mse_loss(d, r))


        ## Part 2:
        # Equation 34
        suff_stat = self.generative_model.layers[0].generate_sufficient_statistic(dreamt_latents[0])
        approx_suff_stat = self.layers[0].approximate_with_alphas(dreamt_data_encodings[0])
        assert approx_suff_stat.shape == suff_stat.shape
        function_approximation_losses = [mse_loss(approx_suff_stat, suff_stat)]

        for i in range(1, len(self.layers)-1): # i = (L - l), so this goes l=L-1,L-2,...,1

            # Equation 32
            mu = self.generative_model.layers[i + 1].mean_sufficient_statistic(dreamt_latents[i])       # E_q[ S_{l-1}(z_{l-1}) | z_l ]
            nabla_g = self.generative_model.layers[i + 1].nabla_natural_parameter(dreamt_latents[i])    # nabla g_{l-1}(z_l)
            mean_stat_gradient = torch.einsum('bn,bnm->bm', mu, nabla_g)
            mean_stat_gradient_approx = self.layers[i].approximate_with_betas(dreamt_data_encodings[i])
            assert mean_stat_gradient_approx.shape == mean_stat_gradient.shape
            mean_stat_gradient_loss = mse_loss(mean_stat_gradient_approx, mean_stat_gradient)

            # Equation 33 XXX: find a way to reuse nabla_g
            sl = self.generative_model.layers[i].generate_sufficient_statistic(dreamt_latents[i])       # S_l(z_l)
            nabla_g = self.generative_model.layers[i].nabla_natural_parameter(dreamt_latents[i - 1])    # nabla g_{l}(z_{l+1})
            higher_stat_gradient = torch.einsum('bn,bnm->bm', sl, nabla_g)
            higher_stat_gradient_approx = self.layers[i].approximate_with_alphas(dreamt_data_encodings[i])
            assert higher_stat_gradient_approx.shape == higher_stat_gradient.shape
            higher_stat_gradient_loss = mse_loss(higher_stat_gradient_approx, higher_stat_gradient)

            function_approximation_losses.append(mean_stat_gradient_loss)
            function_approximation_losses.append(higher_stat_gradient_loss)

        # Equation 31
        nabla_g = self.generative_model.layers[-1].nabla_natural_parameter(dreamt_latents[-2])
        approx_nabla_g = self.layers[-1].approximate_with_alphas(dreamt_data_encodings[-1])
        assert approx_nabla_g.shape == nabla_g.shape
        function_approximation_losses.append(mse_loss(approx_nabla_g, nabla_g))

        return {
            "recognition_losses": recognition_losses,
            "function_approximation_losses": function_approximation_losses,
        }

    
    def wake_phase_gradients(self, data: T) -> List[T]:
        """
        Having trained on gradient function approximation for the sleep phase, it's time
            to apply to approximated gradients to the network parameters, as in equation 10.

        The final parameter increments with approximations are derived in the appendix and
            given in final form in equation 36
        """

        # Encoding approximations r_l, required throughout
        rs = self.full_recognition(data = data)

        # Ordered backwards, as everything else is
        gradients = []

        ## theta_0 update
        gradients.insert(
            0,
            (
                # Approximated nabla g_0 with alphas
                torch.einsum('bn, bnm -> bn', self.generative_model.layers[-1].generate_sufficient_statistic(data), self.layers[-1].approximate_with_alphas(rs[-1]))
                # Approximate muTnabla g_0 with betas from next layer up (away from data - see how betas are trained in sleep_phase_losses)
                - self.layers[-2].approximate_with_betas(rs[-1])
            )
        )
        assert gradients[0].shape[1] == self.generative_model.layers[-1].raw_parameter_values().shape[0]

        for i in range(1, len(self.layers) - 1): # i = (L - l), so this goes l=L-1,L-2,...,1
            
            # theta_l updates - again see how everything are trained in sleep_phase_losses
            gradients.insert(
                0,
                (
                    # Approximate ST nabla g_l with alphas
                    self.layers[i].approximate_with_alphas(rs[i])
                    # Approximate muT nabla g_l with betas
                    - self.layers[i-1].approximate_with_betas(rs[i - 1])
                )
            )
            assert gradients[0].shape[1] == self.generative_model.layers[i].raw_parameter_values().shape[0]

        # theta_L update - requires a logpartition function gradient, as in equation 36
        # This only needs to be implemented for prior distribution layers
        gradients.insert(
            0,
            (
                self.layers[0].approximate_with_alphas(rs[0])
                - self.generative_model.layers[0].log_partition_function_gradient()
            )
        )

        return gradients
