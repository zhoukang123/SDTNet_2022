import torch
import torch.nn.functional as F
import copy
import os
from utils.net_utils import toFloatTensor
#让agent可以知道动作的字符串，也许在未来有作用
class A3CAgent:
    """最简单a3c智能体"""
    def __init__(
        self,
        action_str,
        model,
        gpu_id = -1
    ):
        self.actions = action_str
        self.gpu_id = gpu_id
        self.model = model
        self.done = False
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model = self.model.cuda()

    def model_forward(self, obs, batch_opt = False):
        """obs is dict. values of obs must in numpy, and first dim is batch dim"""
        #TODO 需要unsqueeze,或者重新封装一下单环境
        model_input = obs.copy()#防止obs被改变，因为obs在外部还被保存了一次
        for k in model_input:
            model_input[k] = toFloatTensor(model_input[k], self.gpu_id)
            if not batch_opt:
                model_input[k].unsqueeze_(0)
        out = self.model.forward(model_input)
        return out

    def action(self, env_state):
        out = self.model_forward(env_state)
        pi = out['policy']
        #softmax,形成在动作空间上的分布
        prob = F.softmax(pi, dim = 1).cpu()
        #采样
        action_idx = prob.multinomial(1).item()
        #print(action_idx.shape)
        return self.actions[action_idx], action_idx

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())
        pass

    def save_model(self, path_to_save, title):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        state_to_save = self.model.state_dict()
        import time
        start_time = time.time()
        time_str = time.strftime(
            "%H%M%S", time.localtime(start_time)
        )
        save_path = os.path.join(
            path_to_save,
            "{0}_{1}.dat".format(
                title, time_str
            ),
        )
        torch.save(state_to_save, save_path)
    
    def reset_hidden(self):
        pass
    
    def clear_mems(self):
        pass
