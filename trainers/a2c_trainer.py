from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker
import os

def a2c_train(
    args,
    agent,
    envs,
    runner,
    optimizer,
    loss_func,
    tx_writer
    ):
    #TODO 在a2c中暂时只调用一块gpu用于训练，多线程训练可能需要调用pytorch本身的api
    gpu_id = args.gpu_ids[0]
    
    n_frames = 0
    update_frames = args.nsteps * args.threads
    total_epis = 0
    loss_traker = ScalarMeanTracker()
    pbar = tqdm(total=args.total_train_frames)
    while n_frames < args.total_train_frames:
        
        pi_batch, v_batch, v_final, exps = runner.run()
        loss = loss_func(v_batch, pi_batch, v_final, exps, gpu_id)
        optimizer.zero_grad()
        loss['total_loss'].backward()
        for k in loss:
            loss_traker.add_scalars({k:loss[k].item()})
        optimizer.step()

        #记录、保存、输出
        pbar.update(update_frames)
        n_frames += update_frames
        
        if n_frames % args.print_freq == 0:
            record = runner.pop_mems()
            total_epis += record.pop('epis')
            tx_writer.add_scalar("n_frames", n_frames, total_epis)
            for k,v in record.items():
                tx_writer.add_scalar(k, v, n_frames)
            for k,v in loss_traker.pop_and_reset().items():
                tx_writer.add_scalar(k, v, n_frames)

        if n_frames % args.model_save_freq == 0:
            agent.save_model(args.exp_dir, f'{args.model}_{n_frames}')
    envs.close()
    tx_writer.close()
    pbar.close()