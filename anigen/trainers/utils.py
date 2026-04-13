import torch.nn as nn


def make_master_params_regular(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = []
    for param in model_params:
        if param.requires_grad:
            master_param = nn.Parameter(param.detach().float())
            master_param.requires_grad = True
            master_params.append(master_param)
    return master_params


def model_params_to_master_params_regular(model_params, master_params):
    for param, master_param in zip(model_params, master_params):
        master_param.detach().copy_(param.detach().float())


def master_params_to_model_params_regular(model_params, master_params):
    for param, master_param in zip(model_params, master_params):
        param.detach().copy_(master_param.detach().float())


def model_grads_to_master_grads_regular(model_params, master_params):
    for param, master_param in zip(model_params, master_params):
        if param.grad is not None:
            master_param.grad = param.grad.data.detach().float()
        else:
            master_param.grad = param.data.new_zeros(param.data.shape, dtype=param.data.dtype).float()


# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)


def model_grads_to_master_grads_unsave(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    If a parameter has no gradient, use zeros of the same shape and dtype.
    """
    grads = []
    for param in model_params:
        if param.grad is not None:
            grads.append(param.grad.data.detach().float())
        else:
            grads.append(param.data.new_zeros(param.data.shape, dtype=param.data.dtype).float())
    master_params[0].grad = _flatten_dense_tensors(grads)


def zero_grad(model_params):
    for param in model_params:
       if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()
            

# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0
        