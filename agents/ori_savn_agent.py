import torch
import torch.nn.functional as F
import numpy as np
import copy

from utils.net_utils import toFloatTensor, gpuify
from .a3c_lstm_agent import A3CLstmAgent
#让agent可以知道动作的字符串，也许在未来有作用
class OriSavnAgent:
    """ORI SAVN agent, a3c style"""
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
        self.hidden = [
            gpuify(torch.zeros(1, self.hidden_state_sz), gpu_id),
            gpuify(torch.zeros(1, self.hidden_state_sz), gpu_id)
            ]
        self.probs = gpuify(torch.zeros((1, len(self.actions))), gpu_id)
        self.learned_input = None#注意这个东西在外部重置的，不重置就会一直增加啊
        self.log_pi_batch = []
        self.v_batch = []
        self.entropies = []

    def model_forward(self, obs, batch_opt = False, params = None):

        model_input = obs.copy()
        
        for k in model_input:
            model_input[k] = toFloatTensor(model_input[k], self.gpu_id)
            if not batch_opt:
                model_input[k].unsqueeze_(0)
        model_input['hidden'] = self.hidden
        model_input['action_probs'] = self.probs
        out = self.model.forward(model_input, params)
        out['prob'] = F.softmax(out['policy'], dim = 1)
        out['log_prob'] = F.log_softmax(out['policy'], dim = 1)
        out['entropy'] = (-out['log_prob'] * out['prob']).sum(1)
        return out

    def action(self, env_state, params = None):
        
        out = self.model_forward(env_state, params = params)
        self.probs, self.hidden = out['prob'], out['hidden']
        
        self.v_batch.append(out['value'])
        self.entropies.append(out['entropy'])
        #softmax,形成在动作空间上的分布
        #采样
        action_idx = out['prob'].multinomial(1)

        self.log_pi_batch.append(out['log_prob'].gather(1,action_idx))

        res = torch.cat((self.hidden[0], out['prob']), dim=1)
        if self.learned_input is None:
            self.learned_input = res
        else:
            self.learned_input = torch.cat((self.learned_input, res), dim=0)

        return self.actions[action_idx.cpu().item()], action_idx
    
    def clear_mems(self):
        self.hidden = [
                self.hidden[0].detach(),
                self.hidden[1].detach(),
            ]
        self.probs = self.probs.detach()
        self.log_pi_batch = []
        self.v_batch = []
        self.entropies = []
        self.learned_input = None
    
    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())
        pass

    def reset_hidden(self):
        self.hidden = [
            gpuify(torch.zeros(1, self.hidden_state_sz), self.gpu_id),
            gpuify(torch.zeros(1, self.hidden_state_sz), self.gpu_id)
            ]
        self.probs = gpuify(torch.zeros((1, len(self.actions))), self.gpu_id)
