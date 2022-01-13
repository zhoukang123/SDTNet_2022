import os
os.environ["OMP_NUM_THREADS"] = '1'
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import time
import testers
import models
import agents
import runners
import environment as env
import torch
from tqdm import tqdm
from utils.mean_calc import ScalarMeanTracker, LabelScalarTracker
from utils.thordata_utils import get_scene_names, get_type, get_test_set
from utils.init_func import search_newest_model, get_args, load_or_find_model, make_exp_dir
from utils.record_utils import data_output

def main():
    #读取参数
    args = get_args(os.path.basename(__file__))

    #确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        #torch.cuda.manual_seed(args.seed)
        assert torch.cuda.is_available()
        mp.set_start_method("spawn")

    #动态载入构造函数
    creator = {
        'model':getattr(models, args.model),
        'agent':getattr(agents, args.agent),
        'env':getattr(env, args.env),
    }
    tester = getattr(testers, args.tester)

    model = creator['model'](**args.model_args)
    if model is not None:  print(model)
    #进行一次模型寻找，输出一些调试信息
    load_or_find_model(args)
        
    #这里用于分配各个线程的环境可以加载的场景以及目标
    scene_names_div, chosen_objects, nums_div, test_set_div = get_test_set(args)
    if test_set_div == None:
        test_set_div = [None for _ in range(args.threads)]

    #生成实验文件夹
    make_exp_dir(args, 'TEST')
     #生成各个线程
    processes = []
    result_queue = mp.Queue()
    for i in range(args.threads):
        if args.verbose:
            print('creating threads')
        p = mp.Process(
            target=tester,
            args=(
                args,
                i,
                result_queue,
                args.load_model_dir,
                creator,
                scene_names_div[i],
                chosen_objects,
                nums_div[i],
                test_set_div[i],
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
    print("Test agents created.")

    #取结果并记录
    test_scalars = LabelScalarTracker()

    n_epis = 0

    pbar = tqdm(total = args.total_eval_epi)

    try:
        while n_epis < args.total_eval_epi:

            test_result = result_queue.get()
            n_epis += 1
            
            for k in test_result:
                test_scalars[k].add_scalars(test_result[k])
            pbar.update(1)

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
    pbar.close()

    data_output(args, test_scalars)

if __name__ == "__main__":
    main()

