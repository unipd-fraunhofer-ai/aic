import logging
from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor

from .base_model import BaseModel

from .utils import masked_mean
from ..geometry import Camera, Pose
from ..geometry.optimization import optimizer_step
from ..geometry.interpolation import Interpolator
from ..geometry.costs_multiview import DirectAbsoluteCostMultiview
from ..geometry import losses  # noqa
from ...utils.tools import torchify


logger = logging.getLogger(__name__)

class ClassicMultiviewOptimizer(BaseModel):
    default_conf = dict(
        num_iters=100,
        loss_fn="scaled_barron(-5, 0.5)",
        jacobi_scaling=False,
        normalize_features=False,
        lambda_=1e-2,
        lambda_max=1e4,
        interpolation=dict(
            mode='cubic',
            pad=4,
        ),
        grad_stop_criteria=1e-3,
        dt_stop_criteria=5e-3,  # in meters
        dR_stop_criteria=5e-2,  # in degrees
    )
    logging_fn = None

    def _init(self, conf):
        self.loss_fn = eval('losses.' + conf.loss_fn)
        self.interpolator = Interpolator(**conf.interpolation)
        self.cost_fn = DirectAbsoluteCostMultiview(self.interpolator,
                                          normalize=conf.normalize_features)
        assert conf.lambda_ >= 0.

    def log(self, **args):
        if self.logging_fn is not None:
            self.logging_fn(**args)

    def early_stop(self, **args):
        stop = False
        if not self.training and (args['i'] % 10) == 0:
            T_delta, grad = args['T_delta'], args['grad']
            grad_norm = torch.norm(grad.detach(), dim=-1)
            small_grad = grad_norm < self.conf.grad_stop_criteria
            dR, dt = T_delta.magnitude()
            small_step = ((dt < self.conf.dt_stop_criteria)
                          & (dR < self.conf.dR_stop_criteria))
            if torch.all(small_step | small_grad):
                stop = True
        return stop

    def J_scaling(self, J: Tensor, J_scaling: Tensor, valid: Tensor):
        if J_scaling is None:
            J_norm = torch.norm(J.detach(), p=2, dim=(-2))
            J_norm = masked_mean(J_norm, valid[..., None], -2)
            J_scaling = 1 / (1 + J_norm)
        J = J * J_scaling[..., None, None, :]
        return J, J_scaling

    def build_system(self, J: Tensor, res: Tensor, weights: Tensor):
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weights[..., None] * grad
        grad = grad.sum(-2)  # ... M x 6

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3)  # ... M x 6 x6

        grad = grad.sum(-2) # ... x 6
        Hess = Hess.sum(-3)  # ... x 6 x 6
        return grad, Hess

    def _forward(self, data: Dict):
        return self._run(
            data['p3D'], data['F_ref'], data['F_q'], data['T_init'],
            data['cam_q'], data['mask'], data.get('W_ref_q'))

    @torchify
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)


    def _run(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init_wo: Pose, T_cw: Pose, cameras: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None):
        T = T_init_wo
        J_scaling = None
        cost_sequence = []
        if self.conf.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1) # across feature dim
        args = (T_cw, cameras, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        lambda_ = torch.full_like(failed, self.conf.lambda_, dtype=T.dtype)
        mult = torch.full_like(lambda_, 10)
        recompute = True

        # compute the initial cost
        # mask = mask.flatten() if mask is not None else None
        with torch.no_grad():
            res, valid_i, w_unc_i = self.cost_fn.residuals(T_init_wo, *args)[:3]
            cost_i = self.loss_fn((res.detach()**2).sum(-1))[0]
            if w_unc_i is not None:
                cost_i *= w_unc_i.detach()
            if mask is not None:
                valid_i &= mask
            cost_i_all = masked_mean(cost_i, valid_i, -1) # for each view all valid points
            cost_best = cost_i_all.mean(-1) # together for all views
            cost_sequence.append(cost_i_all)

        for i in range(self.conf.num_iters):
            if recompute:
                res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(
                        T, *args)
                if mask is not None:
                    valid &= mask
                failed = failed | (valid.long().sum(-1).sum(-1) < 10)  # too few points

                cost = (res**2).sum(-1)
                cost, w_loss, _ = self.loss_fn(cost)
                weights = w_loss * valid.float()
                if w_unc is not None:
                    weights *= w_unc
                if self.conf.jacobi_scaling:
                    J, J_scaling = self.J_scaling(J, J_scaling, valid)
                g, H = self.build_system(J, res, weights)

            delta = optimizer_step(g, H, lambda_.unsqueeze(-1), mask=~failed)
            if self.conf.jacobi_scaling:
                delta = delta * J_scaling

            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T_new = T_delta @ T
            # compute the new cost and update if it decreased
            with torch.no_grad():
                res = self.cost_fn.residuals(T_new, *args)[0]
                cost_new = self.loss_fn((res**2).sum(-1))[0]
                cost_new_all = masked_mean(cost_new, valid, -1) # for each view all valid points
                cost_new = cost_new_all.mean(-1) # together for all views
            accept = cost_new < cost_best
            lambda_ = lambda_ * torch.where(accept, 1/mult, mult) # accept shape is (B,)
            lambda_ = lambda_.clamp(max=self.conf.lambda_max, min=1e-7)
            T = Pose(torch.where(accept[..., None], T_new._data, T._data))
            cost_best = torch.where(accept, cost_new, cost_best)
            recompute = accept.any()
            if recompute:
                cost_sequence.append(cost_new_all)

            self.log(i=i, T_init=T_init_wo, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, accept=accept,
                     lambda_=lambda_, H=H, J=J)

            stop = self.early_stop(i=i, T_delta=T_delta, grad=g, cost=cost)
            if self.conf.lambda_ == 0:  # Gauss-Newton
                stop |= (~recompute)
            else:  # LM saturates
                stop |= bool(torch.all(lambda_ >= self.conf.lambda_max))
            if stop:
                break

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed, cost_sequence
    
    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError