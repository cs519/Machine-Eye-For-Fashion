from torch.autograd import Variable
import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, preds, targets):
        # Extract the prections and targets from the tuples
        landmarks_pred, vis_pred = preds
        landmarks_target, vis_target = targets

        # Reshape predictions from 1xX to X
        landmarks_target = landmarks_target.view(1, -1)
        vis_target = vis_target.view(1, -1)

        # Initial the loss functions
        euclid_crit = nn.MSELoss()
        euclid_loss = euclid_crit(landmarks_pred, landmarks_target) / 2

        # cel_crit = nn.CrossEntropyLoss()
        # cel_loss = cel_crit(vis_pred, vis_target)

        # euclid_loss = torch.sum((landmarks_target - landmarks_pred)**2, 2)
        totloss = torch.sum(euclid_loss)#, cel_loss)
        return totloss

    # def __CrossEntropyLoss__(self, pred, targets):
        
