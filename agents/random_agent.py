import torch
import torch.nn.functional as F
import copy
import os
from utils.net_utils import toFloatTensor
import random
#让agent可以知道动作的字符串，也许在未来有作用
class RandomAgent:
    """随机智能体"""
    def __init__(
        self,
        action_str,
        model,
        threads,
        gpu_id = -1
    ):
        self.actions = action_str
        self.gpu_id = gpu_id
        self.model = None
        self.threads = threads
        self.done = False

    def model_forward(self, obs, batch_opt = False):
        
        return None

    def action(self, env_state):
        return [random.choice(self.actions) for _ in range(self.threads)], 0

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        return 

    def save_model(self, path_to_save, title):
        return
    
    def reset_hidden(self, i = 1):
        pass
    
    def clear_mems(self):
        pass
