from torch.nn import ModuleList
from typing import List
from purias_utils.exponential_family_models.base import *


class DeepExponentialFamilyGenerativeModel(nn.Module):
    """
        Layers in top-down order, i.e. z_L first in Vertes and Sahani 2018
    """
    def __init__(self, layers: List[ExponentialFamilyModelLayerBase]) -> None:
        super().__init__()

        self.layers: List[ExponentialFamilyModelLayerBase] = ModuleList(layers)
        assert isinstance(self.layers[0], ExponentialFamilyPriorBase), \
            "First layer in DeepExponentialFamilyModel has to be a ExponentialFamilyPriorBase"

        self.output_dims = [l.output_dim for l in layers]
        self.data_dim = self.output_dims[-1]

    def full_generate(self, batch_size: int):
        layer_outputs = [self.layers[0].sample_conditional_from_previous_latent(batch_size)]
        for layer in self.layers[1:]:
            layer_outputs.append(layer.sample_conditional_from_previous_latent(layer_outputs[-1]))
        return layer_outputs

    def distributed_natural_parameter_generation(self, latents_list: List[_T]):
        "Be careful with ordering here! Reverse order will not throw an error if all layers are the same size!"
        assert len(latents_list) == len(self.layers) - 1
        return [layer.generate_natural_parameter(latent) for layer, latent in zip(self.layers[1:], latents_list)]


class DeepExponentialFamilyRecognitionModel(nn.Module):
    """
        Layers in bottom-up order, i.e. input data in first
    """
    def __init__(self, layers: List[ExponentialFamilyModelLayerBase]) -> None:
        super().__init__()
        self.layers: List[ExponentialFamilyModelLayerBase] = ModuleList(layers)

    def full_generate(self, data: _T):
        layer_outputs = [self.layers[0].sample_conditional_from_previous_latent(data)]
        for layer in self.layers[1:]:
            layer_outputs.append(layer.sample_conditional_from_previous_latent(layer_outputs[-1]))
        return layer_outputs

    def full_generate_natural_parameters(self, data: _T):
        natural_parameter_outputs = [self.layers[0].generate_natural_parameter(data)]
        for i, layer in enumerate(self.layers[1:]):
            z_prev = self.layers[i].sample_conditional_from_natural_parameter(natural_parameter_outputs)
            natural_parameter_outputs.append(layer.generate_natural_parameter(z_prev))
        return natural_parameter_outputs

    def distributed_natural_parameter_generation(self, latents_list: List[_T]):
        "Be careful with ordering here! Reverse order will not throw an error if all layers are the same size!"
        assert len(latents_list) == len(self.layers)
        return [layer.generate_natural_parameter(latent) for layer, latent in zip(self.layers, latents_list)]
