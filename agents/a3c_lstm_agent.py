import torch
import torch.nn.functional as F
import numpy as np
import copy

from utils.net_utils import toFloatTensor, gpuify
#让agent可以知道动作的字符串，也许在未来有作用
class A3CLstmAgent:
    """A3C Lstm agent, a3c style"""
    def __init__(
        self,
        action_str,
        model,
        gpu_id = -1,
        hidden_state_sz = 512
    ):
        self.actions = action_str
        self.gpu_id = gpu_id
        self.model = model
        #self.learned_input = None
        self.hidden_state_sz = hidden_state_sz
        self.done = False#智能体是否提出done？当动作中不含'Done'时，一定一直为False
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model = self.model.cuda()
        
        ###################################################
        self.hidden_batch = [
            gpuify(torch.zeros(1, self.hidden_state_sz), gpu_id),
            gpuify(torch.zeros(1, self.hidden_state_sz), gpu_id)
            ]
        self.probs_batch = gpuify(torch.zeros((1, len(self.actions))), gpu_id)

    def model_forward(self, obs, batch_opt = False):

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
        out = self.model.forward(model_input)
        
        return out

    def action(self, env_state):
        
        out = self.model_forward(env_state)
        pi, _, hidden = out['policy'], out['value'], out['hidden']
        self.hidden_batch[0] = torch.cat((self.hidden_batch[0], hidden[0]), 0)
        self.hidden_batch[1] = torch.cat((self.hidden_batch[1], hidden[1]), 0)
        #softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim = 1).detach()
        self.probs_batch = torch.cat((self.probs_batch, prob), 0)
        #采样
        action_idx = prob.multinomial(1).cpu().item()

        return self.actions[action_idx], action_idx


    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())
        pass

    def reset_hidden(self):
        self.hidden_batch[0][-1] = gpuify(torch.zeros(1, self.hidden_state_sz), self.gpu_id)
        self.hidden_batch[1][-1] = gpuify(torch.zeros(1, self.hidden_state_sz), self.gpu_id)
        self.probs_batch[-1] = gpuify(torch.zeros((1, len(self.actions))), self.gpu_id)
        #self.learned_input = None
    
    def clear_mems(self):
        self.hidden_batch = [
                self.hidden_batch[0][-1:].detach(),
                self.hidden_batch[1][-1:].detach(),
            ]
        self.probs_batch = self.probs_batch[-1:].detach()

