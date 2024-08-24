import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    '''
    if quantile > 0.5 prefered to predict to high (overpredicting)
    if quantile < 0.5 prefered to predict to low (underpredicting)
    '''
    def __init__(self, quantile):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y):
        residual = y - y_pred
        loss = torch.max(residual * (self.quantile - 1), residual * self.quantile)
        return torch.mean(loss)
