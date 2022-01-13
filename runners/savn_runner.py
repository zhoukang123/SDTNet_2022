import numpy as np
import torch
#runner这边要屏蔽torch的相关吗
class SavnRunner:
    """savn runner"""
    def __init__(self, nsteps, threads, env, agent):
        self.rewards = []
        self.masks = []
        self.action_idxs = []
        self.nsteps = nsteps
        self.threads = threads
        self.env = env
        self.agent = agent
        self.done = False

        self.total_epis = 0
        self.total_reward = 0
        self.total_steps = 0
        self.num_success = 0
        self.thread_reward = 0
        self.thread_steps = 0
        self.thread_frames = 0

        self.last_obs = self.env.reset()

    def run(self, params):
        exps = {
            'rewards':[],
            #'masks':[],
            'action_idxs':[]
        }
        #obses = {k:[] for k in self.env.keys}
        #self.agent.clear_mems()
        for _ in range(self.nsteps):
            action, a_idx = self.agent.action(self.last_obs, params)
            obs_new, r, self.done, info = self.env.step(action)
            exps['action_idxs'].append(a_idx)
            exps['rewards'].append(r)
            #exps['masks'].append(1 - done)
            #for k in obses:
                #obses[k].append(self.last_obs[k])
            
            self.thread_reward += r
            self.thread_steps += 1
            self.thread_frames += 1
            
            if self.done:
                self.total_epis += 1
                self.num_success += info['success']
                self.total_reward += self.thread_reward
                self.total_steps += self.thread_steps
                self.thread_steps = 0
                self.thread_reward = 0
                self.last_obs = self.env.reset()
                self.agent.reset_hidden()
                break
            self.last_obs = obs_new
        #if self.done:
            #v_final = 0.0
        #else:
            #model_out = self.agent.model_forward(self.last_obs)
            #v_final = model_out['value'].detach().cpu().item()
        #for k in obses:
            #obses[k] = np.array(obses[k]).reshape(-1, *obses[k][0].shape)
        #out = self.agent.model_forward(obses, True)
        #pi_batch = torch.cat(self.agent.pi_batch, dim = 0)
        #v_batch = torch.cat(self.agent.v_batch, dim = 0)
        return exps
    
    def eval_run(self):
        pass

    def pop_mems(self):
        """will reset all memory, except memory for threads"""
        if self.total_epis == 0 :
            return {
                'epis':0,
                'n_frames':self.thread_frames
                }
        out = {
            'epi length': self.total_steps/self.total_epis,
            'total reward': self.total_reward/self.total_epis,
            "success rate": self.num_success/self.total_epis,
            'epis': self.total_epis,
            'n_frames':self.thread_frames
        }
        self.total_epis = 0
        self.total_reward = 0
        self.total_steps = 0
        self.num_success = 0
        self.thread_frames = 0
        return out

    