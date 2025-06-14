import torch
from torch.optim.optimizer import Optimizer

class Nesterov(Optimizer):
    """Implements Nesterov accelerated gradient.
    This is the same variant of Nesterov momentum as implemented in PyTorch's SGD.
    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super(Nesterov, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Nesterov, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad                
                param_state = self.state[p]

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    buf_old = buf.clone()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    buf_old = buf.clone()

                d_p = d_p.add(buf_old, alpha=momentum)
                p.add_(d_p, alpha=-lr)

        return loss
