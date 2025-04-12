import numpy as np
import torch
import gpytorch
import sys
import warnings



from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel, MaternKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import GammaPrior, NormalPrior


from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.lazy import ConstantDiagLazyTensor, KroneckerProductLazyTensor
from linear_operator.operators import KroneckerProductLinearOperator, DiagLinearOperator


from models.GP_models.transforms import TensorTransform
        
        
class SingleOutputGP(gpytorch.models.ExactGP) :
    
    
    def __init__ (self, train_x, train_y, likelihood, kernel) :
        
        super(SingleOutputGP, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward (self, x) :
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    
    
    
    
class SOGPModel():
    

    is_probabilistic = True
    is_multioutput = False
    is_torch = True

    def __init__(
        self,
        training_max_iter=100,
        learning_rate=0.12,
        input_transform=TensorTransform,
        output_transform=TensorTransform,
        min_loss_rate=None,
        optimizer=None,
        mean=None,
        covar=None,
        show_progress=True,
        silence_warnings=False,
        fast_pred_var=False,
        dev=None,
    ):
        
        self.input_transform = input_transform
        self.output_transform = output_transform

        self.model = None
        self.training_max_iter = training_max_iter
        self.noise_std = None
        self.likelihood = None
        self.mll = None
        self.learning_rate = learning_rate
        self.min_loss_rate = min_loss_rate
        self.mean_module = mean
        self.covar_module = covar
        self.optimizer = optimizer
        self.show_progress = show_progress
        self.predictions = None
        self.fast_pred_var = fast_pred_var
        self.device = dev

        

        if self.optimizer is None:
            warnings.warn("No optimizer specified, using default.", UserWarning)

        # Silence torch and numpy warnings (related to converting
        # between np.arrays and torch.tensors).
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            
    def check_inputs(self, X, y=None):

        # Check input points `X`
        if not isinstance(X, np.ndarray):
            raise ValueError(
                f"Parameters `X` must be of type {np.ndarray} but are"
                f" of type {type(X)} "
            )
        if X.ndim != 2:
            raise ValueError(
                f"Input array `X` must have shape `(n_points, n_features)`"
                f" but has shape {X.shape}."
            )

        if y is not None:
            # Check target `y` is a numpy array
            if not isinstance(y, np.ndarray):
                raise ValueError(
                    f"Targets `y` must be of type {np.ndarray} but are of"
                    f" type {type(y)}."
                )

            # Check shape of `y`
            if y.ndim < 2:
                y = y.reshape( (-1,1))
            if y.ndim > 2 :
                raise ValueError(
                    f"Target array `y` must have at most 2 dimensions and shape "
                    f"(n_points, 1) but has {y.ndim} dimensions and shape "
                    f"{y.shape} "
                )

            # Check consistency of input and output shapes
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Size of input array `X` and output array `y` must match for "
                    f"dimension 0 but are {X.shape[0]} and {y.shape[0]} respectively."
                )

    def create_model(self):

        # Reset optimizer
        self.optimizer = None
        
        dims=len(self.train_X[0])
        
        self.kernel = ScaleKernel(
            RBFKernel(
                has_lengthscale=True, 
                ard_num_dims = dims,
                lengthscale_prior=GammaPrior(1, 10),
                lengthscale_constraint = GreaterThan(5.e-2)
            ),
            outputscale_prior = NormalPrior(3, 0.5)
        )

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        self.likelihood=FixedNoiseGaussianLikelihood(noise = self.noise.reshape(-1) )
              
        self.model = SingleOutputGP(
            self.train_X, self.train_y.reshape((-1,)), self.likelihood, self.kernel
        ).to(self.device)
        
        
        self.model.double()
        self.likelihood.double()
        self.kernel.double()
        
    
    
    def fit(self, train_X, train_y, noise_std = None, **kwargs):
        
        self.check_inputs(train_X, y=train_y)
        train_X = self.input_transform().forward(train_X)
        train_y = self.output_transform().forward(train_y)
        
        
        self.train_X = train_X.to(self.device)
        self.train_y = train_y.to(self.device)
        
        if (noise_std is None) :
            self.noise = None
        else :
            noise = self.input_transform().forward(noise_std**2)
            self.noise = noise.to(self.device)

        # Create model
        self.create_model()

        # Switch the model to train mode
        self.model.train()
        self.likelihood.train()

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_y.reshape((-1,)))
            loss.backward()
            self.vec_loss.append(loss.item())
            self.optimizer.step()

            # TODO: Will this work for negative loss? CHECK
            loss_ratio = None
            if self.min_loss_rate:
                loss_ratio = (loss_0 - loss.item()) - self.min_loss_rate * loss.item()

            # From https://stackoverflow.com/questions/5290994
            # /remove-and-replace-printed-items
            if self.show_progress:
                sys.stdout.write(
                    f"\r Iter = {_i} / {self.training_max_iter},Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                )

            # Get noise value
            self.noise_std = self.model.likelihood.noise.sqrt()

            # Check criterion and break if true
            if self.min_loss_rate:
                if loss_ratio < 0.0:
                    break

            # Set previous iter loss to current
            loss_0 = loss.item()

        print(
            f"Iter = {self.training_max_iter},"
            f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}"
        )
        
        
        
        

    def predict(self, X_pred, return_std=False, **kwargs):
        self.check_inputs(X_pred)
        X_pred = self.input_transform().forward(X_pred)
        
        X_pred = X_pred.to(self.device)
        
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make prediction
        with torch.no_grad(),gpytorch.settings.fast_pred_var(self.fast_pred_var):
            self.prediction = self.model(X_pred)

        # Get mean, variance and std. dev per model
        self.mean =  self.prediction.mean
        self.var =  self.prediction.variance
        self.std =  self.prediction.variance.sqrt()

        # Get confidence intervals per model
        self.cr_l, self.cr_u = self.prediction.confidence_region()

        if return_std:
            return self.output_transform().reverse(self.mean).reshape((-1,1)), self.output_transform().reverse(self.std).reshape((-1,1))
        else:
            return self.output_transform().reverse(self.mean).reshape((-1,1))
        
        
        
        
    def predict_grad(self, X_pred) :
        self.check_inputs(X_pred)
        X_pred = self.input_transform().forward(X_pred)
        
        X = X_pred.to(self.device)

        self.likelihood.eval()
        self.model.eval()
        
        X = torch.tensor(X, requires_grad=True)
        
        self.prediction = self.model(X)
        
        pred = self.prediction.mean
        std = self.prediction.variance.sqrt()
        
        grad = torch.autograd.functional.jacobian(lambda x : self.model(x).mean, X)

        grad=grad.sum(dim=1)
        
        self.mean= pred.detach().numpy()
        self.std=std.detach().numpy()
        self.grad= grad.detach().numpy()
        
        return self.mean, self.std, self.grad
        
    def evaluate_kernel(self, X1, X2) :
        
        X1 = torch.tensor(X1 )
        
        X2 = torch.tensor(X2 )
        
        return self.kernel(X1,X2 ).to_dense().detach().numpy()
        
        
    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)
    
    
    

    def update(self, new_X, new_y, new_noise_std, **kwargs):
        
        self.check_inputs(new_X, y=new_y)
        new_X = self.input_transform().forward(new_X)
        new_y = self.output_transform().forward(new_y)

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)
        

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)
        self.noise = torch.cat([self.noise, new_noise_std**2], dim=0)

        
        self.optimizer = None
        self.fit(self.train_X, self.train_y, self.noise)

        
        
        
        
        