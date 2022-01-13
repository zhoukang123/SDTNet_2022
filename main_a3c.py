#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
#os.environ["MKL_NUM_THREADS"] = '4'
#os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '1'
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import ctypes
import time
import trainers
import models
import agents
import runners
import environment as env
import optimizers
import torch
from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker
from utils.thordata_utils import get_scene_names, random_divide
from utils.init_func import get_args, make_exp_dir
from utils.net_utils import save_model
def main():
    #从命令行读取参数
    args = get_args(os.path.basename(__file__))
    #确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:pass
        #torch.cuda.manual_seed(args.seed)
        #assert torch.cuda.is_available()
        #mp.set_start_method("spawn")


    #载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
        'runner':getattr(runners, args.runner),
        'optimizer':getattr(optimizers, args.optimizer)
    }
    trainer = getattr(trainers, args.trainer)
    loss_func = getattr(trainers, args.loss_func)
    #生成全局模型并初始化优化算法
    shared_model = creator['model'](**args.model_args)
    if shared_model is not None:
        shared_model.share_memory()
        optimizer = None
        if 'Shared' in args.optimizer or 'shared' in args.optimizer:
            optimizer = creator['optimizer'](
                filter(lambda p: p.requires_grad, shared_model.parameters()), **args.optim_args
            )
            optimizer.share_memory()
        print(shared_model)
    # TODO 如果有，读取最新存档模型的参数到shared_model。其余线程会自动使用sync函数来同步
    if args.load_model_dir is not '':
        print("load %s"%args.load_model_dir)
        shared_model.load_state_dict(torch.load(args.load_model_dir))
   
    #这里用于分配各个线程的环境可以加载的场景以及目标
    #这样的操作下每个线程被分配到的场景是不相交的，可以避免读写冲突吧大概
    chosen_scene_names = get_scene_names(args.train_scenes)
    scene_names_div, _ = random_divide(1000, chosen_scene_names, args.threads, args.shuffle)
    chosen_objects = args.train_targets

    #生成实验文件夹
    make_exp_dir(args)
    #初始化TX
    log_writer = SummaryWriter(log_dir = args.exp_dir)

    #生成各个线程
    processes = []
    end_flag = mp.Value(ctypes.c_bool, False)
    result_queue = mp.Queue()

    for thread_id in range(0, args.threads):
        if args.verbose:
            print('creating threads')
        p = mp.Process(
            target=trainer,
            args=(
                args,
                thread_id,
                result_queue,
                end_flag,
                shared_model,
                optimizer,
                creator,
                loss_func,
                scene_names_div[thread_id],
                chosen_objects,  
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Train agents created.")

    #取结果并记录
    print_freq = args.print_freq
    save_freq = args.model_save_freq
    train_scalars = ScalarMeanTracker()

    print_gate_frames = print_freq
    save_gate_frames = save_freq

    n_epis = 0
    n_frames = 0

    pbar = tqdm(total=args.total_train_frames)

    try:
        while n_frames < args.total_train_frames:

            train_result = result_queue.get()
            n_epis += train_result.pop('epis')
            update_frames = train_result.pop('n_frames')
            train_scalars.add_scalars(train_result)
            
            pbar.update(update_frames)
            n_frames += update_frames
            if n_frames >= print_gate_frames:
                print_gate_frames += print_freq
                log_writer.add_scalar("n_epis", n_epis, n_frames)
                tracked_means = train_scalars.pop_and_reset()
                for k, v in tracked_means.items():
                    log_writer.add_scalar(k, v, n_frames)

            if n_frames >= save_gate_frames:
                save_gate_frames += save_freq
                save_model(shared_model, args.exp_dir, f'{args.model}_{n_frames}')

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()
    pbar.close()

if __name__ == "__main__":
    main()

