import numpy as np
#runner这边要屏蔽torch的相关吗
class A2CRunner:
    """生成n steps的数据，同时记录一些数据到自己的变量里"""
    def __init__(self, nsteps, threads, envs, agent):
        self.rewards = []
        self.masks = []
        self.action_idxs = []
        self.nsteps = nsteps
        self.threads = threads
        self.envs = envs
        self.agent = agent

        self.total_epis = 0
        self.total_reward = 0
        self.total_steps = 0
        self.num_success = 0
        self.thread_reward = [0 for _ in range(threads)]
        self.thread_steps = [0 for _ in range(threads)]

        self.last_obs = envs.reset()

    def run(self):
        exps = {
            'rewards':[],
            'masks':[],
            'action_idxs':[]
        }
        obses = {k:[] for k in self.envs.keys}
        self.agent.clear_mems()
        for _ in range(self.nsteps):
            action, a_idx = self.agent.action(self.last_obs)
            obs_new, r, done, info = self.envs.step(action)
            for k in obses:
                obses[k].append(self.last_obs[k])
            exps['action_idxs'].append(a_idx)
            exps['rewards'].append(r)
            exps['masks'].append(1 - done)
            self.last_obs = obs_new
            self.total_epis += done.sum()
            for i in range(self.threads):
                self.thread_reward[i] += r[i]
                self.thread_steps[i] += 1
                self.num_success += info[i]['success']
                if done[i]:
                    self.total_reward += self.thread_reward[i]
                    self.total_steps += self.thread_steps[i]
                    self.thread_steps[i] = 0
                    self.thread_reward[i] = 0
                    self.agent.reset_hidden(i)
        out = self.agent.model_forward(self.last_obs)
        v_final = out['value'].detach().cpu().numpy().reshape(-1)
        for k in obses:
            obses[k] = np.array(obses[k]).reshape(-1, *obses[k][0][0].shape)
        out = self.agent.model_forward(obses, True)
        pi_batch, v_batch = out['policy'], out['value']
        return pi_batch, v_batch, v_final, exps
    
    def eval_run(self):
        pass

    def pop_mems(self):
        """will reset all memory, except memory for threads"""
        if self.total_epis == 0 :
            return {'epis':0}
        out = {
            'epi length': self.total_steps/self.total_epis,
            'total reward': self.total_reward/self.total_epis,
            "success rate": self.num_success/self.total_epis,
            'epis': self.total_epis
        }
        self.total_epis = 0
        self.total_reward = 0
        self.total_steps = 0
        self.num_success = 0
        return out

    