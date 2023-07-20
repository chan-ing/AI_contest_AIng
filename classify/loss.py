import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

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
# class LovaszLoss(nn.Module):
#     def __init__(self):
#         super(LovaszLoss, self).__init__()

#     def lovasz_grad(self, gt_sorted):
#         gts = gt_sorted.sum()
#         intersection = gts - gt_sorted.float().cumsum(dim=0)
#         union = gts + (1 - gt_sorted).float().cumsum(dim=0)
#         jaccard = 1.0 - intersection / union
#         jaccard[1:] = jaccard[1:] - jaccard[:-1]
#         return jaccard

#     def lovasz_hinge_flat(self, logits, targets):
#         targets = targets.view(-1)
#         logits = logits.view(-1)
#         signs = 2.0 * targets - 1.0
#         errors = 1.0 - logits * signs
#         errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
#         perm = perm.data
#         gt_sorted = targets[perm]
#         grad = self.lovasz_grad(gt_sorted)
#         loss = torch.dot(torch.nn.functional.relu(errors_sorted), grad)
#         return loss

#     def forward(self, inputs, targets):
#         losses = []
#         num_classes = inputs.size(1)
#         for c in range(num_classes):
#             target_c = (targets == c).float()
#             input_c = inputs[:, c, ...]
#             loss_c = self.lovasz_hinge_flat(input_c, target_c)
#             losses.append(loss_c)
#         loss = torch.mean(torch.stack(losses))
#         return loss


# Lovasz loss 클래스 수정 구현
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors

def lovasz_hinge_flat(logits, labels, ignore_index):
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore_index is not None:
        mask = labels != ignore_index
        logits = logits[mask]
        labels = labels[mask]
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss

class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)


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
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_score
        return loss
    
#loss = criterion(outputs, masks)