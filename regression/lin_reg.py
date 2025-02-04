import torch 


class LinReg(torch.nn.Module):

    def __init__(self, n_features):
        super(LinReg, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
      out = self.lr(x) 
      return out