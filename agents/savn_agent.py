import torch
import torch.nn.functional as F
import numpy as np
import copy

from utils.net_utils import toFloatTensor, gpuify
from .a3c_lstm_agent import A3CLstmAgent
#让agent可以知道动作的字符串，也许在未来有作用
class SavnAgent(A3CLstmAgent):
    """SAVN agent, a3c style"""
    def __init__(
        self,
        action_str,
        model,
        gpu_id = -1,
        hidden_state_sz = 512
    ):
        super(SavnAgent, self).__init__(
            action_str,
            model,
            gpu_id,
            hidden_state_sz
            )
        self.learned_input = None#注意这个东西在外部重置的，不重置就会一直增加啊
        self.pi_batch = []
        self.v_batch = []

    def model_forward(self, obs, batch_opt = False, params = None):

        model_input = obs.copy()
        
        for k in model_input:
            model_input[k] = toFloatTensor(model_input[k], self.gpu_id)
            if not batch_opt:
                model_input[k].unsqueeze_(0)
        if batch_opt:
            model_input['hidden'] = (
                self.hidden_batch[0][:-1],
                self.hidden_batch[1][:-1],
                )
            model_input['action_probs'] = self.probs_batch[:-1]
        else:
            model_input['hidden'] = (
                self.hidden_batch[0][-1:],
                self.hidden_batch[1][-1:],
                )
            model_input['action_probs'] = self.probs_batch[-1:]
        out = self.model.forward(model_input, params)
        
        return out

    def action(self, env_state, params = None):
        
        out = self.model_forward(env_state, params = params)
        pi, v, hidden = out['policy'], out['value'], out['hidden']
        self.pi_batch.append(pi)
        self.v_batch.append(v)
        self.hidden_batch[0] = torch.cat((self.hidden_batch[0], hidden[0]), 0)
        self.hidden_batch[1] = torch.cat((self.hidden_batch[1], hidden[1]), 0)
        #softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim = 1).detach()
        self.probs_batch = torch.cat((self.probs_batch, prob), 0)
        #采样
        action_idx = prob.multinomial(1).cpu().item()

        res = torch.cat((self.hidden_batch[0][-1:], prob), dim=1)
        if self.learned_input is None:
            self.learned_input = res
        else:
            self.learned_input = torch.cat((self.learned_input, res), dim=0)

        return self.actions[action_idx], action_idx
    
    def clear_mems(self):
        self.hidden_batch = [
                self.hidden_batch[0][-1:].detach(),
                self.hidden_batch[1][-1:].detach(),
            ]
        self.probs_batch = self.probs_batch[-1:].detach()
        self.pi_batch = []
        self.v_batch = []
        self.learned_input = None
