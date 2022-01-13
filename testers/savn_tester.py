import time
import random
import torch
from trainers.train_util import get_params, transfer_gradient_to_shared, SGD_step
from utils.mean_calc import ScalarMeanTracker
from utils.env_wrapper import SingleEnv
from utils.thordata_utils import get_type

def savn_test(
    args,
    thread_id,
    result_queue,
    load_model_dir,
    creator,
    chosen_scene_names,
    chosen_objects,
    t_epis,
    test_sche = None,
    gradient_limit = 4
):

    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #设置随机数种子
    torch.manual_seed(args.seed + thread_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + thread_id)
    #initialize env and agent

    model = creator['model'](**args.model_args)
    if load_model_dir is not None:
        model.load_state_dict(torch.load(load_model_dir))
    model.eval()

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
        # theta <- shared_initialization
        params_list = [get_params(model, gpu_id)]
        params = params_list[-1]
        #loss_dict = {}
        episode_num = 0
        num_gradients = 0
        agent.reset_hidden()
        agent.clear_mems()
        last_obs = env.reset(**test_sche[t_epis])
        # Accumulate loss over all meta_train episodes.
        if args.verbose:
            print("#########################")
            print(f'in {env.env.scene_name} towards {env.env.target_str}')
        done = False
        thread_reward = 0
        thread_steps = 0
        while True:
            # Run episode for k steps or until it is done or has made a mistake (if dynamic adapt is true).
            agent.learned_input = None
            
            for _ in range(args.nsteps):
                action, _ = agent.action(last_obs, params)
                if args.verbose:
                    print(action)
                obs_new, r, done, info = env.step(action)
            
                thread_reward += r
                thread_steps += not info['agent_done']
                #thread_steps += 1

                if done:
                    spl = 0
                    if info['success']:
                        assert info['best_len'] <= thread_steps,f"{info['best_len']}!={thread_steps}"
                        spl = info['best_len']/thread_steps
                    data = {
                        'ep_length:':thread_steps,
                        'SR:':info['success'],
                        'SPL:':spl,
                        'total_reward:':thread_reward,
                        'epis':1
                    }
                    if args.verbose:
                        print(thread_steps,info['best_len'],spl)
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

            if gradient_limit < 0 or episode_num < gradient_limit:

                num_gradients += 1
                learned_loss = agent.model.learned_loss(
                        agent.learned_input, params
                    )
                agent.learned_input = None

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss,
                    [v for _, v in params_list[episode_num].items()],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                params_list.append(
                    SGD_step(params_list[episode_num], inner_gradient, args.inner_lr)
                )
                params = params_list[-1]

                episode_num += 1

        


