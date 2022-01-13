import time
import random
import torch
from trainers.train_util import get_params, transfer_gradient_to_shared, SGD_step
from utils.mean_calc import ScalarMeanTracker
from utils.env_wrapper import SingleEnv
from utils.thordata_utils import get_type

def a3c_test(
    args,
    thread_id,
    result_queue,
    load_model_dir,
    creator,
    chosen_scene_names,
    chosen_objects,
    t_epis,
    test_sche = None,
):

    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #设置随机数种子
    torch.manual_seed(args.seed + thread_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + thread_id)
    #initialize env and agent

    model = creator['model'](**args.model_args)
    if load_model_dir is not '':
        model.load_state_dict(torch.load(load_model_dir))

    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        gpu_id
    )
    env = creator['env'](
        offline_data_dir = args.offline_data_dir,
        action_dict = args.action_dict,
        target_dict = args.target_dict,
        obs_dict = args.obs_dict,
        reward_dict = args.reward_dict,
        max_steps = args.max_epi_length,
        grid_size = args.grid_size,
        rotate_angle = args.rotate_angle,
        chosen_scenes = chosen_scene_names,
        chosen_targets = chosen_objects
    )
    env = SingleEnv(env, True)
    if test_sche == None:
        test_sche = []
        for _ in range(t_epis):
            test_sche.append(dict(
                scene_name = None, 
                target_str = None, 
                agent_state = None, 
            ))
    else:
        for i in range(t_epis):
            test_sche[i] = dict(
                scene_name = test_sche[i][0], 
                target_str = test_sche[i][1], 
                agent_state = test_sche[i][2], 
                )
        test_sche.reverse()
    while t_epis:
        t_epis -= 1

        agent.reset_hidden()
        agent.clear_mems()
        last_obs = env.reset(**test_sche[t_epis])
        done = False
        thread_reward = 0
        thread_steps = 0
        while True:
            if args.verbose:
                print("New inner step")
            
            for _ in range(args.nsteps):
                action, _ = agent.action(last_obs)
                obs_new, r, done, info = env.step(action)
            
                thread_reward += r
                thread_steps += not info['agent_done']

                if done:
                    spl = 0
                    if info['success']:
                        assert info['best_len'] <= thread_steps
                        spl = info['best_len']/thread_steps
                    data = {
                        'ep_length:':thread_steps,
                        'SR:':info['success'],
                        'SPL:':spl,
                        'total_reward:':thread_reward,
                        'epis':1
                    }
                    target_str = get_type(info['scene_name'])+'/'+info['target']
                    res = {
                        info['scene_name']:data,
                        target_str:data
                    }
                    result_queue.put(res)
                    break
                last_obs = obs_new
            
            if done:
                break

        


