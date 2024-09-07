import torch
from torch import Tensor as _T

from tqdm import tqdm
from typing import List, Union

from purias_utils.geometry.topology import Points
from purias_utils.geometry.embedding import Embedding


class EmbeddedTangentFieldTargettedRNN:

    def __init__(self, embedding: Embedding, non_lin: str = 'tanh', scale: float = 1) -> None:
        
        self.embedding = embedding
        self.manifold = embedding.domain
        ungained_non_lin = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "linear": lambda x: x,
        }[non_lin]
        self.non_lin = lambda x: ungained_non_lin(x / scale)

        ## Constructions that remain the same regardless of tangent field
        # First, make a filter for all the gridpoints that are charted to NaNs
        self.point_connectivty = torch.cat([
            chart.domain.contains(self.manifold.grid_points)
            for chart in self.manifold.charts
        ], 0)

        # Repeat x for each chart, then remove NaNs (out of domain)
        x = torch.cat([self.embedding.embedded_grid_points.coords for _ in self.manifold.charts])
        self.phi_ps = x[self.point_connectivty]
        self.x = self.non_lin(self.phi_ps)

        x_pinv = torch.linalg.inv(x.T @ x)
        x_pinv += torch.eye(x_pinv.shape[0]) * 1e-6 
        self.x_pinv = (x_pinv @ x.T)

    def ols_fit_w_rec(self, embedded_tangent_vectors: List[_T]):
        assert len(embedded_tangent_vectors) == len(self.manifold.charts)
        y = torch.cat(embedded_tangent_vectors, 0)
        y = y[self.point_connectivty]
        x = self.x
        return {
            "W_OLS": (self.x_pinv @ y).detach(),
            "PHI_PS": self.phi_ps,
            "X_PINV": self.x_pinv,
            "Y": y
        }

    def generate_trajectory(self, W: _T, initial_position: Union[_T, Points], dt: float, num_steps: int):
        u_history = [initial_position if isinstance(initial_position, T) else initial_position.coords]

        for t in tqdm(range(num_steps - 1)):
            r = self.non_lin(u_history[-1])
            du_dt = r @ W
            u = u_history[-1] + du_dt * dt
            u_history.append(u)

        u_history = torch.stack(u_history, 1)

        return u_history 



class SecondOrderEmbeddedTangentFieldTargettedRNN(EmbeddedTangentFieldTargettedRNN):

    def __init__(
        self, embedding: Embedding, non_lin: str = 'tanh', scale: float = 1
    ) -> None:
        super().__init__(embedding, non_lin, scale)

    def ols_fit_w_rec(self, embedded_tangent_vectors: List[_T], embedded_tangent_vector_tangent_vectors: List[_T]):

        first_order_results = super().ols_fit_w_rec(embedded_tangent_vectors)
        second_order_results = super().ols_fit_w_rec(embedded_tangent_vector_tangent_vectors)

        first_order_results.update(
            {
                "Y_DOT": second_order_results["Y"],
                "Z_OLS": second_order_results["W_OLS"]
            }
        )
        return first_order_results

    def generate_trajectory(self, W: _T, Z: _T, initial_position: Union[_T, Points], dt: int, num_steps: int):
        u_history = [initial_position if isinstance(initial_position, T) else initial_position.coords]

        total_dynamics_matrix = (dt * W) + (0.5 * dt * Z)

        for t in tqdm(range(num_steps - 1)):
            r = self.non_lin(u_history[-1])
            du = r @ total_dynamics_matrix
            u = u_history[-1] + du
            u_history.append(u)

        u_history = torch.stack(u_history, 1)

        return u_history 


