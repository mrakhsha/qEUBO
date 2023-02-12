import torch

from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor

from src.models.likelihoods.preferential_softmax_likelihood import (
    PreferentialSoftmaxLikelihood,
)


class PreferentialVariationalGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        use_withening: bool = True,
    ) -> None:
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]
        train_x = queries.reshape(queries.shape[0] * queries.shape[1], queries.shape[2])
        train_y = responses.squeeze(-1)
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)], dtype=torch.double
        ).T
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds, n=2 * self.input_dim, q=1
            ).squeeze(1)
            inducing_points = torch.cat([inducing_points, train_x], dim=0)
            # Construct variational dist/strat
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        else:
            inducing_points = train_x
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        super().__init__(variational_strategy)
        self.likelihood = PreferentialSoftmaxLikelihood(num_points=self.q)
        # Mean and cov
        self.mean_module = ConstantMean()
        scales = bounds[1, :] - bounds[0, :]

        use_pairwise_gp_covar = False
        if use_pairwise_gp_covar:
            os_lb, os_ub = 1e-2, 1e2
            ls_prior = GammaPrior(1.2, 0.5)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            self.covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=ls_prior,
                    lengthscale_constraint=GreaterThan(
                        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
                    ),
                ),
                outputscale_prior=SmoothedBoxPrior(a=os_lb, b=os_ub),
                # make sure we won't get extreme values for the output scale
                outputscale_constraint=Interval(
                    lower_bound=os_lb * 0.5,
                    upper_bound=os_ub * 2.0,
                    initial_value=1.0,
                ),
            )
        else:
            self.covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0 / scales),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
