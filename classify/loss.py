import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

# Focal loss 구현
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

# Lovasz loss 클래스 구현
class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_hinge(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        signs = 2. * targets.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = targets[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def lovasz_softmax(self, probas, targets):
        C = probas.size(1)
        losses = []
        for c in range(C):
            target_c = (targets == c).float()
            if probas.size(1) == 1:
                probas_c = probas[:, 0]
            else:
                probas_c = probas[:, c]
            losses.append(self.lovasz_hinge(probas_c, target_c))
        mean_loss = torch.mean(torch.stack(losses))
        return mean_loss

    def forward(self, probas, targets):
        return self.lovasz_softmax(probas, targets)

# Dice loss 클래스 구현
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
#loss = criterion(outputs, masks)

# DiceBCE loss 클래스 구현
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
#loss = criterion(outputs, masks.unsqueeze(1))


# CrossEntropy+Lovasz loss 함수
class BceAndLovasz(nn.Module):
    def __init__(self):
        super(BceAndLovasz, self).__init__()
        self.bce_weight = 0.8
        self.class_bce_weights = torch.tensor([0.5]) 
        self.lovasz_weight = 0.2
        self.class_lovasz_weights = torch.tensor([1.0]) 
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.lovasz_loss = LovaszLoss()

    def forward(self, inputs, targets):
        # CrossEntropy Loss
        bce_loss_value = self.bce_loss(inputs, targets)
        bce_loss_value = torch.sum(bce_loss_value * self.class_bce_weights)

        # Compute Lovasz Loss
        lovasz_loss_value = self.lovasz_loss(inputs, targets)
        lovasz_loss_value = torch.sum(lovasz_loss_value * self.class_lovasz_weights)

        # Combine the losses with given weights
        combined_loss = self.bce_weight * bce_loss_value + self.lovasz_weight * lovasz_loss_value

        return combined_loss



# 복합 손실 함수 정의
class CombinedLoss(nn.Module):
    def __init__(self, loss_functions, loss_weights):
        super(CombinedLoss, self).__init__()
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights

    def forward(self, inputs, targets):
        combined_loss = 0
        for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
            loss_value = loss_fn(inputs, targets)
            combined_loss += weight * loss_value

        return combined_loss

# loss_function1 = DiceLoss()
# loss_function2 = LovaszLoss()

# loss_functions = [loss_function1, loss_function2]
# loss_weights = [0.8, 0.2]

# combined_loss_function = CombinedLoss(loss_functions, loss_weights)
# criterion = combined_loss_function
