
class GaborGSM(nn.Module):
    def __init__(self, n_filters: int, square_size: int, scale_lower: float, scale_upper: float, device = 'cuda') -> None:

        self.n_filters = n_filters
        self.square_size = square_size

        assert (scale_lower, scale_upper) == (0.2, 1.0) # For now

        super(GaborGSM, self).__init__()

        thetas = torch.rand(self.n_filters)
        scales = (torch.rand(self.n_filters) * (scale_upper - scale_lower)) + scale_lower
        x_mids = (
            torch.rand(self.n_filters) * self.square_size - (self.square_size / 2)
        )
        y_mids = (
            torch.rand(self.n_filters) * self.square_size - (self.square_size / 2)
        )

        log_pixel_noise_std = torch.tensor(0.0)
        log_contrast_alpha = torch.tensor(log(2.0))
        log_contrast_beta = torch.tensor(log(0.5))

        # Completely overwritten later, this is just for loading state dict
        latent_prior_covar_cholesky_untriled = torch.zeros(n_filters, n_filters)

        self.register_parameter(name='thetas', param=Parameter(thetas.to(device)))
        self.register_parameter(name='scales', param=Parameter(scales.to(device)))
        self.register_parameter(name='x_mids', param=Parameter(x_mids.to(device)))
        self.register_parameter(name='y_mids', param=Parameter(y_mids.to(device)))
        self.register_parameter(name='log_pixel_noise_std', param=Parameter(log_pixel_noise_std.to(device)))
        self.register_parameter(name='log_contrast_alpha', param=Parameter(log_contrast_alpha.to(device)))
        self.register_parameter(name='log_contrast_beta', param=Parameter(log_contrast_beta.to(device)))
        self.register_parameter(name='latent_prior_covar_cholesky_untriled', param=Parameter(latent_prior_covar_cholesky_untriled.to(device)))

    @property
    def latent_prior_covar_cholesky(self):
        "Allows easy calling of the C^L"
        return torch.tril(self.latent_prior_covar_cholesky_untriled)

    @property
    def pixel_noise_std(self):
        return self.log_pixel_noise_std.exp()

    @property
    def contrast_alpha(self):
        return self.log_contrast_alpha.exp()

    @property
    def contrast_beta(self):
        return self.log_contrast_beta.exp()

    @property
    def frozen_projective_fields(self):

        return self.frozen_filter_set.reshape(self.square_size * self.square_size, -1)

    @property
    def frozen_filter_set(self):

        projective_fields, _, _ = gabor(
            theta=self.thetas,
            scale=self.scales,
            _x=self.x_mids,
            _y=self.y_mids,
            square_size=self.square_size
        )

        return projective_fields.detach()
        


