import torch.nn as nn


class NormCrossEntropyLoss(object):
    def __init__(self):

        self.loss_op = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y)
        loss = (loss * data.node_norm)[data.train_mask].sum()
        return loss


class NormBCEWithLogitsLoss(object):
    def __init__(self):
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y)
        loss = (loss * data.node_norm)[data.train_mask].sum()
        return loss



