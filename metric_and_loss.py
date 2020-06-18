import torch.nn as nn
import torch


class NormCrossEntropyLoss(object):
    def __init__(self):
        self.loss_op = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.long())
        loss = (loss * data.node_norm)[data.train_mask].sum()
        return loss


class NormBCEWithLogitsLoss(object):
    def __init__(self):
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.type_as(out))

        # loss = (loss * data.node_norm)[data.train_mask].sum()
        loss = torch.mul(loss.T, data.node_norm).T[data.train_mask].sum()
        return loss


class FixedBCEWithLogitsLoss(object):
    def __init__(self,reduction='none'):
        self.loss_op = nn.BCEWithLogitsLoss(reduction=reduction)

    def __call__(self, out, data):
        return self.loss_op(out, data.type_as(out))
