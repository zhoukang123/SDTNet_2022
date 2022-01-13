"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
import ctypes
from collections import OrderedDict

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict

def split_data_info(info_list):
    """
    Input the info of obs in tuple:[(key,shape,dtype)]
    exp:[
            ('fc',(1,2048),np.float32),
            ('score',(1,1000),np.float32),
            ...
    ]
    """
    keys = []
    shapes = {}
    dtypes = {}
    for key in info_list:
        shape_, dtype_ = info_list[key]
        keys.append(key)
        shapes[key] = shape_
        dtypes[key] = dtype_
    return keys, shapes, dtypes

_NP_TO_CT = {np.dtype(np.float32): ctypes.c_float,
             np.dtype(np.int32): ctypes.c_int32,
             np.dtype(np.int8): ctypes.c_int8,
             np.dtype(np.uint8): ctypes.c_char,
             np.dtype(np.bool): ctypes.c_bool}

def make_envs(env_args, env_class):
    """预封装环境的生成函数，为生成多线程环境服务的"""
    def _env_func():
        return env_class(**env_args)
    return _env_func

class SingleEnv:
    """对单线程环境的再封装，目的是为了和vecenv统一数据(把target和obs塞一起了)以及实现测试序列"""
    def __init__(self, env, eval_mode = False):
        self.env = env
        self.eval_mode = eval_mode
        self.t_reper = None
        self.keys, self.shapes, self.dtypes = split_data_info(env.data_info)

    def reset(self, **kwargs):
        obs, self.t_reper = self.env.reset(calc_best_len = self.eval_mode, **kwargs)
        obs.update(self.t_reper)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs.update(self.t_reper)
        return obs, reward, done, info

    def get_obs(self):
        obs = self.env.get_obs()
        obs.update(self.t_reper)
        return obs

    def get_target_reper(self):
        return self.t_reper

class VecEnv:
    closed = False
    def __init__(self, env_fns, context='spawn', eval_mode = False, test_sche = None):
        """
        多线程环境。对env的一个封装，输入的是环境的构造函数.eval_model下会及算最短路，很慢。
        test_sche不为None时可以进行测试序列。
        """
        ctx = mp.get_context(context)
        env = env_fns[0]()
        self.keys, self.shapes, self.dtypes = split_data_info(env.data_info)
        self.obs_keys = env.obs_info.keys()
        self.t_keys = env.target_reper_info.keys()
        env.close()
        del env
        self.num_envs = len(env_fns)
        if test_sche == None:
            test_sche = [[] for _ in range(len(env_fns))]
        self.data_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.dtypes[k]], int(np.prod(self.shapes[k]))) for k in self.keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        thread = -1
        for env_fn, data_buf in zip(env_fns, self.data_bufs):
            parent_pipe, child_pipe = ctx.Pipe()
            env_fn = CloudpickleWrapper(env_fn)
            thread += 1
            proc = ctx.Process(target=_subproc_worker,
                        args=(
                            child_pipe, 
                            parent_pipe, 
                            env_fn, 
                            data_buf, 
                            self.shapes, 
                            self.dtypes,
                            eval_mode,
                            test_sche[thread]
                            ))
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            #logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        [pipe.recv() for pipe in self.parent_pipes]
        return self.get_obs()

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        _, rews, dones, info = zip(*outs)
        return self.get_obs(), np.array(rews), np.array(dones), info
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()
    
    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def get_target_reper(self):
        result = {}
        for k in self.t_keys:

            bufs = [b[k] for b in self.data_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.dtypes[k]).reshape(self.shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)

    def get_obs(self):
        """把target reper 和obs从buf里全读出来"""
        result = {}
        for k in self.keys:

            bufs = [b[k] for b in self.data_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.dtypes[k]).reshape(self.shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)


def _subproc_worker(pipe, parent_pipe, env_fn, bufs, obs_shapes, obs_dtypes, eval_mode, test_sche = []):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    if test_sche is not []:
        for i in range(len(test_sche)):
            test_sche[i] = dict(
                scene_name = test_sche[i][0], 
                target_str = test_sche[i][1], 
                agent_state = test_sche[i][2], 
                )
    def _write_bufs(dict_data):
        for k in dict_data:
            dst = bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, dict_data[k])
    env = env_fn.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                if test_sche == []:
                    obs, t_reper = env.reset(calc_best_len = eval_mode)
                else:
                    obs, t_reper = env.reset(calc_best_len = eval_mode,**test_sche.pop())
                pipe.send((_write_bufs(obs), _write_bufs(t_reper)))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    if test_sche == []:
                        obs, t_reper = env.reset(calc_best_len = eval_mode)
                    else:
                        obs, t_reper = env.reset(calc_best_len = eval_mode,**test_sche.pop())
                    _write_bufs(t_reper)
                pipe.send((_write_bufs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render())
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
