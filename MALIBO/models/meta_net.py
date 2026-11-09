import torch
import torch.nn as nn

class MetaLearnerNet(nn.Module):
    def __init__(self, input_dim=1, embed_dim=4, hidden_dim=64):
        super(MetaLearnerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim + embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if z.dim() == 1:
            z = z.repeat(x.size(0), 1)
        elif z.size(0) != x.size(0):
            z = z.repeat(x.size(0), 1)
            
        inp = torch.cat([x, z], dim=-1)
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        return self.sigmoid(self.out(h))