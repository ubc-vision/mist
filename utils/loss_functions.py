import torch

def one_hot_classification_loss(logits, labels_gt, num_classes, loss_type='MSE', power=2):
    diag = {}
    logits = torch.mean(logits, dim=1)
    logits_gt = torch.nn.functional.one_hot(labels_gt, num_classes).float()
    logits_gt = torch.mean(logits_gt, dim=1)
    loss = torch.nn.MSELoss(reduction='none')(logits, logits_gt)
    loss = loss.sum(dim=1) /2
    diag['loss_per_sample'] = loss.detach()
    return loss.mean(), diag

