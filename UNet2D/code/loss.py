import torch
from torch import nn, Tensor
import torch.nn.functional as F


def dice_loss(predict, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(predict * target)
    dice = (2 * intersect + smooth) / (torch.sum(target * target) + torch.sum(predict * predict) + smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        # input: [N, 1, H, W]
        N, _, H, W = input_tensor.shape
        one_hot = torch.zeros((N, self.n_classes, H, W), device=input_tensor.device)
        one_hot.scatter_(1, input_tensor.long(), 1)
        return one_hot


    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            inputs = F.softmax(input, dim=1)
        target = self.one_hot_encode(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        loss = loss / self.n_classes
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss


class DiceCeLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0):
        '''
        calculate:
            celoss + alpha*celoss
            alpha : set def to 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)

    def forward(self, predict, label):

        diceloss = self.diceloss(predict, label)
        celoss = self.celoss(predict, label)
        loss = celoss + self.alpha * diceloss
        return loss
