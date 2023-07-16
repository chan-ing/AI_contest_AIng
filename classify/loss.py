import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-bce_loss)  # 확률 값 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, logits, targets):
        return self.lovasz_hinge(logits, targets)

    def lovasz_hinge(self, logits, targets):
        signs = 2 * targets - 1
        errors = (1 - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.squeeze()
        gt_sorted = targets[perm]
        grad = self.lovasz_grad(gt_sorted)

        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.float()
        intersection = gts.sum()
        union = gts.numel()
        grad = torch.zeros_like(gts)

        for i in range(1, len(gts)):
            grad[i] = (intersection - gts[:i].sum()) / (union - gts[:i].numel())

        return grad