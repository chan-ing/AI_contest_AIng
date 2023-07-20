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
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(dim=0)
        union = gts + (1 - gt_sorted).float().cumsum(dim=0)
        jaccard = 1.0 - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def lovasz_hinge_flat(self, logits, targets):
        targets = targets.view(-1)
        logits = logits.view(-1)
        signs = 2.0 * targets - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = targets[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(torch.nn.functional.relu(errors_sorted), grad)
        return loss

    def forward(self, inputs, targets):
        losses = []
        num_classes = inputs.size(1)
        for c in range(num_classes):
            target_c = (targets == c).float()
            input_c = inputs[:, c, ...]
            loss_c = self.lovasz_hinge_flat(input_c, target_c)
            losses.append(loss_c)
        loss = torch.mean(torch.stack(losses))
        return loss

# 복합 손실 함수 정의
class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, lovasz_weight=0.5):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lovasz_weight = lovasz_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(gamma=self.gamma)
        self.lovasz_loss = LovaszLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        lovasz_loss = self.lovasz_loss(inputs, targets)

        compound_loss = (self.alpha * bce_loss) + ((1 - self.alpha) * focal_loss) + (self.lovasz_weight * lovasz_loss)
        return compound_loss

#------------------아래와 같이 main에서 사용----------------------#

# 복합 손실 함수와 optimizer 정의
# criterion = CompoundLoss(alpha=0.5, gamma=2, lovasz_weight=0.5)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # training loop
# for epoch in range(10):  # 10 에폭 동안 학습합니다.
#     model.train()
#     epoch_loss = 0
#     for images, masks in tqdm(dataloader):
#         images = images.float().to(device)
#         masks = masks.float().to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks.unsqueeze(1))
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

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