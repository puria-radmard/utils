import torch
from torch import nn
from torch import Tensor as T
from torch.nn.functional import binary_cross_entropy, sigmoid

from typing import List

from purias_utils.exponential_family_models.bernoulli import LinearBernoulliPriorModelLayer, LinearBernoulliModelLayer
from purias_utils.exponential_family_models.zdeep import DeepExponentialFamilyGenerativeModel, DeepExponentialFamilyRecognitionModel


class BinaryHelmholtz(nn.Module):
    """
        This is the H.M. described in Dayan and Abbott, chapter 10, when latent_dims is length 1
    """
    def __init__(self, data_dim: int, latent_dims: List[int]):

        super(BinaryHelmholtz, self).__init__()

        self.data_dim = data_dim
        self.latent_dims = latent_dims
        self.num_latent_layers = len(latent_dims)

        generative_model_layers = [LinearBernoulliPriorModelLayer(latent_dims[0])]
        for i in range(1, len(latent_dims)):
            generative_model_layers += [LinearBernoulliModelLayer(latent_dims[i], latent_dims[i+1])]
        generative_model_layers += [LinearBernoulliModelLayer(latent_dims[-1], data_dim)]

        recongition_model_layers = []
        for i in range(1, len(latent_dims)):
            recongition_model_layers += [LinearBernoulliModelLayer(latent_dims[i+1], latent_dims[i])]
        recongition_model_layers += [LinearBernoulliModelLayer(data_dim, latent_dims[0])]
        recongition_model_layers = recongition_model_layers[::-1]

        self.generative_model = DeepExponentialFamilyGenerativeModel(generative_model_layers)
        self.recognition_model = DeepExponentialFamilyRecognitionModel(recongition_model_layers)

    def draw_data(self, batch_size: int):
        "Draw from P[u|v,G] as required in the sleep phase"
        return self.generative_model.full_generate(batch_size=batch_size)

    def draw_latent(self, data: T):
        "Draw from P[v|u,W] as required in the wake phase"
        return self.recognition_model.full_generate(data=data)

    def wake_phase_losses(self, data_minibatch: T):
        """
            Draw z from each layer conditioned on sampled data, 
            then pass each through the relevant layer in the generative model
        """
        # Ensure recognition model W is kept constant by detaching
        with torch.no_grad():
            recognised_latents = self.recognition_model.full_generate(data=data_minibatch)
        generated_bernoullis = self.generative_model.distributed_natural_parameter_generation(
            latents_list = recognised_latents[::-1]
        )
        targets_list = recognised_latents[1:] + [data_minibatch]
        return [binary_cross_entropy(g, t) for g, t in zip(generated_bernoullis, targets_list)]

    def sleep_phase_losses(self, batch_size: int):
        """
            Hallucinate latents and data, and pass them backwards ("rerecognise them") and compare using BCE

            Basically a consistency check between forward and backward models
        """
        # Ensure generative model G is kept constant by detaching
        with torch.no_grad():
            dreamt_latents = self.generative_model.full_generate(batch_size=batch_size)
        rerecognised_latents = self.recognition_model.full_generate_natural_parameters(dreamt_latents[-1])
        return [binary_cross_entropy(r, d) for r, d in zip(rerecognised_latents, dreamt_latents)]



class BinaryHelmholtzEasyMode(nn.Module):
    """
        A more interpretable implementaiton of the H.M. described in Dayan and Abbott, chapter 10!
    """
    def __init__(self, data_dim: int, latent_dim: int):

        super(BinaryHelmholtzEasyMode, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim

        # Generative model (approximation)
        self.G_and_h = nn.Linear(latent_dim, data_dim, bias = True)
        self.register_parameter(name='g', param=torch.nn.Parameter(torch.randn(latent_dim)))
        self.g: torch.nn.Parameter

        # Recognition model
        self.W_and_w = nn.Linear(data_dim, latent_dim, bias = True)

    def latent_prior_bernoulli(self, batch_size, logit = False):
        "Eq 10.41"
        logit_result = self.g.unsqueeze(0).repeat(batch_size, 1)
        return logit_result if logit else sigmoid(logit_result)

    def generative_model_conditional_bernoulli(self, latent_vs: T, logit = False):
        "Eq 10.42"
        assert latent_vs.shape[-1] == self.latent_dim
        logit_result = self.G_and_h(latent_vs)
        return logit_result if logit else sigmoid(logit_result)
    
    def recognition_model_variational_bernoulli(self, data_us: T, logit = False):
        "Eq 10.43"
        assert data_us.shape[-1] == self.data_dim
        logit_result = self.W_and_w(data_us)
        return logit_result if logit else sigmoid(logit_result)

    def draw_data(self, bernoulli: T):
        "Draw from P[u|v,G] as required in the sleep phase"
        assert bernoulli.shape[-1] == self.data_dim
        return torch.bernoulli(bernoulli)

    def draw_latent(self, bernoulli: T):
        "Draw from P[v|u,W] as required in the wake phase"
        assert bernoulli.shape[-1] == self.latent_dim
        return torch.bernoulli(bernoulli)

    def wake_phase_losses(self, minibatch: T):
        """
            Provide a batch of u (drawn from the real distribution, i.e. the real dataset),
            and draw a latent variable from the current recognition distribution.

            MSE losses will induce the updates provided in the chapter appendix
        """
        # Ensure recognition model W is kept constant by detaching
        with torch.no_grad():
            rec_conditional = self.recognition_model_variational_bernoulli(minibatch)
            latent = self.draw_latent(rec_conditional)
        g_loss = binary_cross_entropy(self.latent_prior_bernoulli(latent.shape[0]), latent)
        G_and_h_loss = binary_cross_entropy(self.generative_model_conditional_bernoulli(latent), minibatch)

        return g_loss, G_and_h_loss, latent

    def sleep_phase_losses(self, batch_size):
        """
            This time, hallucinate a latent-data pair
        """
        # Ensure generative model model G is kept constant by detaching
        with torch.no_grad():
            prior = self.latent_prior_bernoulli(batch_size)
            dreamt_latent = self.draw_latent(prior)

            gen_conditional = self.generative_model_conditional_bernoulli(dreamt_latent)
            dreamt_data = self.draw_data(gen_conditional)

        rerecognised_latent_conditional = self.recognition_model_variational_bernoulli(dreamt_data)

        return binary_cross_entropy(rerecognised_latent_conditional, dreamt_latent), rerecognised_latent_conditional, dreamt_latent, dreamt_data



