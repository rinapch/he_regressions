import torch



class LogReg(torch.nn.Module):

    def __init__(self, n_features):
        super(LogReg, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out