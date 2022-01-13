import torch
import torch.nn as nn
import torch.nn.functional as F

class DemoModel(nn.Module):
    def __init__(self, action_size):
        super(DemoModel, self).__init__()
        self.state_embed = nn.Linear(2048, 512)
        self.hidden_fc = nn.Linear(512, 128)
        self.pi = nn.Linear(128, action_size)
        self.v = nn.Linear(128, 1)
    def forward(self, model_input):
        x = self.state_embed(model_input['fc'])
        x = F.relu(x)
        x = self.hidden_fc(x)
        pi = self.pi(F.relu(x))
        v = self.v(F.relu(x))
        return dict(policy=pi, value=v)