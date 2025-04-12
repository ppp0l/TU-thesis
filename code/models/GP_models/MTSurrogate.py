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
from gpytorch.priors import GammaPrior


from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.lazy import ConstantDiagLazyTensor, KroneckerProductLazyTensor
from linear_operator.operators import KroneckerProductLinearOperator, DiagLinearOperator

from models.GP_models.transforms import TensorTransform
from models.surrogate import Surrogate


        
        
        
        
        
        
class FixedNoiseMultitaskLikelihood(_MultitaskGaussianLikelihoodBase):
    
    def __init__(self, noise, num_tasks,
                 *args, 
                 **kwargs):
        
        self.noise= noise
        noise_covar = FixedGaussianNoise(noise=noise)
        super().__init__(noise_covar=noise_covar, num_tasks= num_tasks, *args, **kwargs)
        
        
        self.has_global_noise = True
        self.has_task_noise = False
        
        
        
    def _shaped_noise_covar(self, shape, add_noise=True, *params, **kwargs):
        
        data_noise = self.noise_covar(*params, shape =(shape[0],), **kwargs)
        
        eye = torch.ones(self.num_tasks, device=data_noise.device, dtype=data_noise.dtype)
        
        task_noise = DiagLinearOperator(
            eye,
        )
        
        return KroneckerProductLinearOperator(data_noise, task_noise)
    
    

        
        

    

class IndipendentMultiTaskGP(gpytorch.models.ExactGP):
    """
    From: https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Multitask_GP_Regression.html # noqa: E501
    """

    def __init__(self, train_x, train_y, likelihood, N_tasks, kernel):
        super(IndipendentMultiTaskGP, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.MultitaskMean( gpytorch.means.ConstantMean() , N_tasks)
        
        self.covar_module = gpytorch.kernels.MultitaskKernel( kernel, num_tasks= N_tasks, rank=0 )
                                                             

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultitaskMultivariateNormal(mean_x, covar_x)


    

class MTModel(Surrogate):

    def __init__(
        self,
        num_tasks=None,
        training_max_iter=200,
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
        nu=0,
    ):
        
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.n_batches = None
        self.n_features = None
        self.n_training = None

        self.model = None
        self.num_tasks = num_tasks
        self.dout = num_tasks
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
        self.nu = nu
        

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
                raise ValueError(
                    f"Target array `y` must have at least 2 dimensions and shape "
                    f"(n_points, n_outputs) but has {y.ndim} dimensions and shape "
                    f"{y.shape} "
                )

            # Check consistency of input and output shapes
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Size of input array `X` and output array `y` must match for "
                    f"dimension 0 but are {X.shape[0]} and {y.shape[0]} respectively."
                )

    def create_model(self,):

        if self.num_tasks is None:
            self.num_tasks = self.train_y.shape[1]

        # Reset optimizer
        self.optimizer = None

        # Check input consistency
        if self.train_y.shape[0] != self.train_X.shape[0]:
            raise ValueError(
                f"Dim 0 of `train_y` must be equal to the number of training"
                f"samples but is {self.train_y.shape[0]} != {self.train_X.shape[0]}."
            )

        if self.train_y.shape[1] != self.num_tasks:
            raise ValueError(
                f"Dim 1 of `train_y` must be equal to the number of tasks"
                f"but is {self.train_y.shape[1]} != {self.num_tasks}"
            )
            
        if (self.noise is None) :
            self.likelihood = MultitaskGaussianLikelihood(
                num_tasks=self.num_tasks
            ).to(self.device)
        
        
        else :
            #self.likelihood= FixedNoiseGaussianLikelihood(
               # noise=self.noise, #batch_shape=self.num_tasks
            #).to(self.device)
            # currently doesn't work
            # setup is that the covariance is `D \kron I` where `D` is user supplied
            self.likelihood = FixedNoiseMultitaskLikelihood(num_tasks=self.num_tasks, noise=self.noise, rank=0)

        
        dims=self.train_X.shape[1]
        lengthscale = 0.15
        
        if self.nu != 0:
            self.kernel= MaternKernel(nu=self.nu, 
                             has_lengthscale=True, 
                             ard_num_dims = dims,
                             lengthscale_prior=GammaPrior(1, 10),
                             lengthscale_constraint = GreaterThan(5.e-2)
                             )
        else :
            self.kernel=RBFKernel(
                     has_lengthscale=True,
                     ard_num_dims = dims,
                     lengthscale_prior=GammaPrior(1, 10),
                     lengthscale_constraint = GreaterThan(5.e-2)
                     )
                
            
                
        self.model = IndipendentMultiTaskGP(
            self.train_X, self.train_y, self.likelihood, self.num_tasks, self.kernel
        ).to(self.device)
        
        #self.model.covar_module.data_covar_module.register_constraint("raw_lengthscale", GreaterThan(1.e-3))
        #self.model.covar_module.data_covar_module.register_constraint("raw_lengthscale", Interval(1.e-3, lengthscale))
        
        self.model.double()
        self.likelihood.double()
        self.kernel.double()
    
    
    def fit(self, train_X, train_y, noise_std = None, **kwargs):
        self.check_inputs(train_X, y=train_y)
        self.dim = len(train_X[0])
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
        
        #dims=len(self.train_X[0])
        
        # init_lengthscale = 0.4
        
        # self.model.covar_module.data_covar_module.lengthscale = init_lengthscale
        
        all_params = set(self.model.parameters())

        # Define optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                all_params, lr=self.learning_rate
            )  # Includes GaussianLikelihood parameters

        # Define marginal loglikelihood
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Train
        self.vec_loss = []
        loss_0 = np.inf
        for _i in range(self.training_max_iter):

            self.optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_y)
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
                print(
                    f"Iter = {_i} / {self.training_max_iter},"
                    f" Loss = {loss.item()}, Loss_ratio = {loss_ratio}",
                    end="\r",
                    flush=True,
                )

            # Get noise value
            #self.noise_std = self.model.likelihood.noise.sqrt()

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
            return self.output_transform().reverse(self.mean), self.output_transform().reverse(self.std)
        else:
            return self.output_transform().reverse(self.mean)
        
        
    def predict_grad(self, X_pred) :

        self.check_inputs(X_pred)
        X_pred = self.input_transform().forward(X_pred)

        X_pred = X_pred.to(self.device)

        self.likelihood.eval()
        self.model.eval()
        
        self.prediction = self.model(X_pred)
        
        pred = self.prediction.mean
        std = self.prediction.variance.sqrt()
        
        grad = torch.autograd.functional.jacobian(lambda x : self.model(x).mean, X_pred)
        
        grad=grad.sum(dim=2)
        
        self.mean= pred.detach().numpy()
        self.std=std.detach().numpy()
        self.grad= grad.detach().numpy()
        
        return pred.detach().numpy(), std.detach().numpy(), grad.detach().numpy()
    
    
    def evaluate_kernel(self, X1, X2) :
        
        X1 = torch.tensor(X1 )
        
        X2 = torch.tensor(X2 )
        
        return self.kernel(X1, X2 ).to_dense().detach().numpy()
        
        
        
    def sample_posterior(self, n_samples=1):
        # Switch the model to eval mode
        self.model.eval()
        self.likelihood.eval()

        return self.prediction.n_sample(n_samples)
    
    
    

    def update(self, new_X, new_y, new_noise_std=None, updated_y = None, updated_noise = None):

        self.check_inputs(new_X, y=new_y)
        new_X = self.input_transform().forward(new_X)
        new_y = self.output_transform().forward(new_y)

        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)

        if updated_y is not None:
            updated_y = self.output_transform().forward(updated_y)
            train_y = updated_y.to(self.device)
        else:
            train_y = self.train_y

        if updated_noise is not None:
            updated_noise = self.input_transform().forward(updated_noise)
            noise = updated_noise.to(self.device)
        else:
            noise = self.noise        

        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_y = torch.cat([train_y, new_y], dim=0)
        
        if (new_noise_std is None) or (self.noise is None):
            self.noise=None
        else :
            new_noise = self.input_transform().forward(new_noise_std**2)
            new_noise = new_noise.to(self.device)
            
            self.noise = torch.cat([noise, new_noise], dim=0)

        
        self.optimizer = None
        self.fit(self.train_X, self.train_y, noise = self.noise)

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
