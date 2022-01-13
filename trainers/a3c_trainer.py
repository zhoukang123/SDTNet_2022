import torch
from .train_util import copy_gradient
from torch.nn.utils import clip_grad_norm_
from utils.mean_calc import ScalarMeanTracker
from utils.env_wrapper import SingleEnv
#import setproctitle

def a3c_train(
    args,
    thread_id,
    result_queue,
    end_flag,#多线程停止位
    shared_model,
    optim,
    creator,
    loss_func,
    chosen_scene_names = None,
    chosen_objects = None,
):
    #setproctitle.setproctitle("Training Agent: {}".format(thread_id))
    #判断是否有gpu,分配gpu
    if args.verbose:
        print('agent %s created'%thread_id)
    
    gpu_id = args.gpu_ids[thread_id % len(args.gpu_ids)]
    torch.cuda.set_device(gpu_id)
    #设置随机数种子
    torch.manual_seed(args.seed + thread_id)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + thread_id)
    #initialize env and agent

    model = creator['model'](**args.model_args)
    if args.verbose:
        print('model created')
    agent = creator['agent'](
        list(args.action_dict.keys()),
        model,
        gpu_id
    )
    if args.verbose:
        print('agent created')
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
    env = SingleEnv(env, False)
    #initialize a episode
    runner = creator['runner'](
        args.nsteps, 1, env, agent
    )
    if optim == None:
        optim = creator['optimizer'](
            shared_model.parameters(),
            **args.optim_args
        )

    #n_frames = 0
    #total_epis = 0
    #print_freq = args.print_freq / args.threads
    #update_frames = args.nsteps
    loss_tracker = ScalarMeanTracker()
    while not end_flag.value:
        
        # Train on the new episode.
        agent.sync_with_shared(shared_model)
        # Run episode for num_steps or until player is done.
        pi_batch, v_batch, v_final, exps = runner.run()
        if args.verbose:
            print('Got exps')
        # Compute the loss.
        loss = loss_func(v_batch, pi_batch, v_final, exps, gpu_id=gpu_id)
        loss["total_loss"].backward()
        for k in loss:
            loss_tracker.add_scalars({k:loss[k].item()})
            
        if args.verbose:
            print('Loss computed')

        clip_grad_norm_(model.parameters(), 50.0)

        # Transfer gradient to shared model and step optimizer.
        copy_gradient(shared_model, model)
        optim.step()
        model.zero_grad()
            
        if args.verbose:
            print('optimized')
        
        #n_frames += update_frames
        #if n_frames % print_freq == 0:
        results = runner.pop_mems()
        #total_epis += record['epis']
        #results.update(dict(n_frames=print_freq))
        results.update(loss_tracker.pop_and_reset())
        result_queue.put(results)
