import math
from typing import Literal

import torch
import torch.nn.functional as F
from autoattack.autopgd_base import APGDAttack
from torch import Tensor


class APGDTargeted(APGDAttack):
    def __init__(
        self,
        predict: torch.nn.Module,
        eps: float,
        n_iter: int = 100,
        norm: Literal['L2', 'L2', 'Linf'] = 'Linf',
        loss: Literal['ce-targeted', 'dlr-targeted'] = 'ce-targeted',
        use_largereps: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            predict, 
            n_iter, 
            norm,
            eps=eps, 
            seed=seed, 
            loss=loss,
            use_largereps=use_largereps, 
        )

    def dlr_loss_targeted(self, x: Tensor, y: Tensor) -> Tensor:
        x_sorted = x.sort(dim=1)[0]
        u = torch.arange(x.shape[0])
        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
               x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def ce_loss_targeted(self, x: Tensor, _: Tensor) -> Tensor:
        return - F.cross_entropy(x, self.y_target, reduction='none')
    
    def perturb(self, x: Tensor, y: Tensor) -> Tensor:
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y = y.detach().clone().long().to(self.device)

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed) # type: ignore

        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig, .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])

        adv_best = x.detach().clone()
        loss_best = torch.full((len(x),), -float('inf'), device=self.device)

        self.y_target = y
        
        for _ in range(self.n_restarts):

            if not self.use_largereps:
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
            else:
                best_curr, _, loss_curr, _ = self.decr_eps_pgd(x, y, epss, iters)

            ind_curr = (loss_curr > loss_best).nonzero().squeeze()

            adv_best[ind_curr] = best_curr[ind_curr] # type: ignore
            loss_best[ind_curr] = loss_curr[ind_curr]

        return adv_best