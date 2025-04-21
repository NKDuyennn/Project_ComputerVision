from lib import *

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10  # To avoid division by zero

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        # L2Norm: Chuan hoa de tinh ra mot vector co do dai bang 1
        # x.size() = (batch_size, channels, height, width) channels:dim=1
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps  # (batch_size, 1, height, width)
        x = torch.div(x, norm)

        # weights.size() = (512)
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)  # (batch_size, 512, height, width)
        x = torch.mul(x, weight)

        return x
    
