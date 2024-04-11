from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.func import functional_call, jacrev
from tqdm import tqdm
import torch.nn as nn
import torch


class Subnet:
    def __init__(
        self, model: nn.Module, snapshot_freq=1, count_params_subnet=300
    ) -> None:
        self.snapshot_freq = 1
        self.model = model
        self.param_vector_model = self._param_vector(self.model)
        self.mean = torch.zeros_like(self.param_vector_model)
        self.sq_mean = torch.zeros_like(self.param_vector_model)
        self.n = 1
        self.count_params_subnet = count_params_subnet
        self.min_var = 1e-30

    @staticmethod
    def _param_vector(model: nn.Module):
        return parameters_to_vector(model.parameters())

    def update_model_variances(self, epoch, total_epochs):
        if epoch % self.snapshot_freq != 0:
            return
        old_fac, new_fac = self.n / (self.n + 1), 1 / (self.n + 1)
        self.mean = self.mean * old_fac + self._param_vector(self.model) * new_fac
        self.sq_mean = (
            self.sq_mean * old_fac + self._param_vector(self.model) ** 2 * new_fac
        )
        self.n += 1

    def get_mask_indices(self):
        param_variances = torch.clamp(self.sq_mean - self.mean**2, self.min_var)
        idx = torch.argsort(param_variances, descending=True)[
            : self.count_params_subnet
        ]
        idx = idx.sort()[0]
        parameter_vector = self._param_vector(self.model)
        subnet_mask = torch.zeros_like(parameter_vector).bool()
        subnet_mask[idx] = 1
        return subnet_mask.nonzero(as_tuple=True)[0]


class Laplace:
    def __init__(self, model: nn.Module, device: torch.device, subnet: Subnet) -> None:
        self.device = device
        self.model = model
        self.params_dict = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }
        self.buffers_dict = {k: v for k, v in model.named_buffers()}
        self.params_dict = {
            key: value.to(self.device) for key, value in self.params_dict.items()
        }
        self.buffers_dict = {
            key: value.to(self.device) for key, value in self.buffers_dict.items()
        }
        self.subnet = subnet
        self.subnet_mask_indices = subnet.get_mask_indices()
        self.H = torch.zeros(
            subnet.count_params_subnet, subnet.count_params_subnet, device=device
        )
        self.prior_precision = 1.0
        temperature = 1.0
        sigma_noise = 1.0
        sigma_noise = torch.tensor(sigma_noise, device=device)
        sigma2 = sigma_noise.square()
        self.H_factor = 1 / (sigma2 * temperature)
        self.posterior_covariance = None

    def _jacobian(self, data):
        def _model_fn_params_only(params_dict, buffers_dict):
            out = functional_call(self.model, (params_dict, buffers_dict), data)
            return out, out

        with torch.no_grad():
            Js, f = jacrev(_model_fn_params_only, has_aux=True)(
                self.params_dict, self.buffers_dict
            )

        Js = [
            j.flatten(start_dim=-p.dim())
            for j, p in zip(Js.values(), self.params_dict.values())
        ]
        Js = torch.cat(Js, dim=-1)
        Js = Js[:, :, self.subnet_mask_indices]
        return Js, f

    def calculate_ggn(self, data_loader, lossfn):
        self.loss = 0
        self.H = torch.zeros(
            self.subnet.count_params_subnet,
            self.subnet.count_params_subnet,
            device=self.device,
        )
        for X, y in tqdm(data_loader):
            self.model.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
            Js, f = self._jacobian(X)
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum("mk,mc->mck", ps, ps)
            H_batch = torch.einsum("bcp,bck,bkq->pq", Js, H_lik, Js)
            loss_batch = 1.0 * lossfn(f, y)
            self.loss += loss_batch
            self.H += H_batch

        prior_precision_diag = torch.ones(
            self.subnet.count_params_subnet, device=self.device
        )
        posterior_precision = self.H_factor * self.H + torch.diag(prior_precision_diag)
        invsqrt_precision = _precision_to_scale_tril
        posterior_scale = invsqrt_precision(posterior_precision)
        self.posterior_covariance = posterior_scale @ posterior_scale.T

    def calculate_ppd(self, X):
        Js, f_mu = self._jacobian(X)
        Js = Js.squeeze(0)
        f_var = torch.einsum("np,pq,mq->nm", Js, self.posterior_covariance, Js)
        f_var = f_var.unsqueeze(0)
        kappa = 1 / torch.sqrt(1.0 + torch.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
        final_ppd = torch.softmax(kappa * f_mu, dim=-1)
        return final_ppd
