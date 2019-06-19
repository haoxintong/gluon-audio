# MIT License
# Copyright (c) 2019 haoxintong
"""Custom losses.
Losses are subclasses of gluon.loss.SoftmaxCrossEntropyLoss which is a HybridBlock actually."""
import math
import numpy as np
from mxnet import nd
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

__all__ = ["get_loss", "SoftmaxCrossEntropyLoss", "ArcLoss", "RingLoss",
           ]
numeric_types = (float, int, np.generic)


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is nd.ndarray else F.reshape_like(x, y)


class ArcLoss(SoftmaxCrossEntropyLoss):
    r"""ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.


    - Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, classes, m=0.5, s=64, easy_margin=True, dtype="float32", **kwargs):
        super().__init__(**kwargs)
        assert s > 0.
        assert 0 <= m < (math.pi / 2)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self._classes = classes
        self.easy_margin = easy_margin
        self._dtype = dtype

    def hybrid_forward(self, F, pred, label, sample_weight=None, *args, **kwargs):
        cos_t = F.pick(pred, label, axis=1)  # cos(theta_yi)
        if self.easy_margin:
            cond = F.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - self.threshold
            cond = F.Activation(data=cond_v, act_type='relu')

        # sin_t = F.sqrt(1.0 - cos_t * cos_t)  # sin(theta)
        # new_zy = cos_t * self.cos_m - sin_t * self.sin_m  # cos(theta_yi + m)

        new_zy = F.cos(F.arccos(cos_t) + self.m)  # cos(theta_yi + m)
        if self.easy_margin:
            zy_keep = cos_t
        else:
            zy_keep = cos_t - self.mm  # (cos(theta_yi) - sin(pi - m)*m)
        new_zy = F.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t  # cos(theta_yi + m) - cos(theta_yi)
        diff = F.expand_dims(diff, 1)  # shape=(b, 1)
        gt_one_hot = F.one_hot(label, depth=self._classes, on_value=1.0, off_value=0.0, dtype=self._dtype)
        body = F.broadcast_mul(gt_one_hot, diff)
        pred = pred + body
        pred = pred * self.s

        return super().hybrid_forward(F, pred=pred, label=label, sample_weight=sample_weight)


class RingLoss(SoftmaxCrossEntropyLoss):
    r"""Computes the Ring Loss from
    `"Ring loss: Convex Feature Normalization for Face Recognition"
    <https://arxiv.org/abs/1803.00130>`_ paper.

    .. math::
        L = -\sum_i \log \softmax({pred})_{i,{label}_i} +  \frac{\lambda}{2m} \sum_{i=1}^{m}
         (\Vert \mathcal{F}({x}_i)\Vert_2 - R )^2

    Parameters
    ----------
    lamda: float
        The loss weight enforcing a trade-off between the softmax loss and ring loss.


    - Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.

    """

    def __init__(self, lamda, weight_initializer=None, dtype='float32', **kwargs):
        super().__init__(**kwargs)

        self._lamda = lamda
        self.R = self.params.get('R', shape=(1,), init=weight_initializer,
                                 dtype=dtype, allow_deferred_init=True)

    def hybrid_forward(self, F, pred, label, embedding, R, sample_weight=None):
        # RingLoss
        emb_norm = F.norm(embedding, axis=1)
        loss_r = F.square(F.broadcast_sub(emb_norm, R)) * 0.5
        loss_r = _apply_weighting(F, loss_r, self._weight, sample_weight)

        # Softmax
        loss_sm = super().hybrid_forward(F, pred, label, sample_weight)

        return loss_sm + self._lamda * loss_r


_losses = {
    'softmax': SoftmaxCrossEntropyLoss,
    'arcface': ArcLoss,
    'ringloss': RingLoss,
}


def get_loss(name, **kwargs):
    """
    Parameters
    ----------
    name : str
        Name
    kwargs : str
        Params
    Returns
    -------
    HybridBlock
        The loss.
    """
    name = name.lower()
    if name not in _losses:
        err_str = '"%s" is not among the following losses list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_losses.keys())))
        raise ValueError(err_str)
    loss = _losses[name](**kwargs)
    return loss


def get_loss_list():
    """Get the entire list of loss names in losses.
    Returns
    -------
    list of str
        Entire list of loss names in losses.
    """
    return _losses.keys()
