"""Autograd helper for repeated derivatives in BINN PDE losses.

The `Gradient` function computes first- or higher-order derivatives by
recursively differentiating with respect to the same input tensor.
"""

from torch.autograd import grad

def Gradient(outputs, inputs, order=1):
    """Differentiate `outputs` with respect to `inputs` up to `order` times.

    Parameters
    ----------
    outputs : torch.Tensor
        Tensor expression to differentiate.
    inputs : torch.Tensor
        Differentiation variable(s) tracked by autograd.
    order : int, optional
        Derivative order. `order=1` computes first derivatives.

    Returns
    -------
    torch.Tensor
        The derivative tensor after `order` recursive applications.
    """
    
    # return outputs if derivative order is 0
    grads = outputs
    
    # convert outputs to scalar
    outputs = outputs.sum()

    # compute gradients sequentially until order is reached
    for i in range(order):
        grads = grad(outputs, inputs, create_graph=True)[0]
        outputs = grads.sum()

    return grads
