import torch.nn as nn
from .gcn_base import GcnBaseModel
from .tcn import TemporalConvNet


class GCNSAVN(GcnBaseModel):
    def __init__(
        self,
        action_sz,
        nsteps,
        target_sz = 300,
        dropout_rate = 0.25,
        ):
        super(GCNSAVN, self).__init__(
            action_sz, 
            target_sz,
            dropout_rate,
            )

        self.feature_size = 512 + action_sz
        self.learned_input_sz = 512 + action_sz

        self.num_steps = nsteps
        #self.ll_key = nn.Linear(self.feature_size, self.feature_size)
        #self.ll_linear = nn.Linear(self.feature_size, self.feature_size)
        self.ll_tc = TemporalConvNet(
            self.num_steps, [10, 1], kernel_size=2, dropout=0.0
        )

    def learned_loss(self, H, params=None):
        H_input = H.unsqueeze(0)
        x = self.ll_tc(H_input, params).squeeze(0)
        return x.pow(2).sum(1).pow(0.5)
