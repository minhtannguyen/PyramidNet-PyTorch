# coding: utf-8

import torch
from torch.autograd import Function


class ShakeFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, bdrop):
        ctx.save_for_backward(x, alpha, beta, bdrop)

        y = x * (bdrop + alpha - bdrop * alpha)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, bdrop = ctx.saved_variables
        grad_x = grad_alpha = grad_beta = grad_bdrop = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bdrop + beta - bdrop * beta)

        return grad_x, grad_alpha, grad_beta, grad_bdrop


shake_function = ShakeFunction.apply


def get_alpha_beta_bdrop(batch_size, droprate, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = 2.0*torch.rand(1) - 1.0
    elif forward_shake and shake_image:
        alpha = 2.0*torch.rand(batch_size).view(batch_size, 1, 1, 1) - 1.0
    else:
        alpha = torch.FloatTensor([0.0])

    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])
        
    if backward_shake and not shake_image:
        bdrop = torch.bernoulli(torch.FloatTensor([droprate]))
    elif backward_shake and shake_image:
        bdrop = torch.bernoulli(droprate*torch.ones([batch_size, 1, 1, 1]))
    else:
        bdrop = torch.FloatTensor([droprate])

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
        bdrop = bdrop.cuda()

    return alpha, beta, bdrop
