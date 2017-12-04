import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.euclid_crit = nn.MSELoss()
        self.cel_crit = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # Extract the prections and targets from the tuples
        landmarks_pred, vis_pred = preds
        landmarks_target, vis_target = targets

        # Reshape predictions from 1xX to X
        landmarks_target = landmarks_target.view(1, -1)
        vis_target = vis_target.view(1, -1)

        # Initial the loss functions
        euclid_loss = self.euclid_crit(landmarks_pred, landmarks_target) / 2

        # cel_loss = self.cel_crit(vis_pred, vis_target)

        # euclid_loss = torch.sum((landmarks_target - landmarks_pred)**2, 2)
        totloss = torch.sum(euclid_loss)# , cel_loss)
        return totloss

    def __CrossEntropyLoss__(self, pred, targets):
        r, c = targets.size()

        total_loss = 0

        for i in range(r):
            for j in range(0, c, 3):
                total_loss += self.cel_crit(pred[i][j:j + 3],
                                            targets[i][j:j + 3])

        return total_loss
