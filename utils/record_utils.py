from .mean_calc import LabelScalarTracker
from .thordata_utils import get_type
import os

def data_output(args, test_scalars):
    """整理数据并输出到json。输入的这个是一个dict，键有房间名称，
    以及前缀一个场景类型的目标字符串(为了告诉函数这个目标是在哪个房间找的)"""
    total_scalars = LabelScalarTracker()
    scene_split = {k:{} for k in args.test_scenes}
    target_split = {k:{} for k in args.test_scenes}
    result = test_scalars.pop_and_reset(['epis'])

    for k in result:
        k_sp = k.split('/')
        if len(k_sp) == 1:
            s_type = get_type(k_sp[0])
            scene_split[s_type][k] = result[k].copy()
            total_scalars[s_type].add_scalars(result[k])
            total_scalars['Total'].add_scalars(result[k])
        else:
            target_split[k_sp[0]][k_sp[-1]] = result[k].copy()
            total_scalars[k_sp[-1]].add_scalars(result[k])
    
    total_scalars = total_scalars.pop_and_reset(['epis'])
    
    import json
    for k in scene_split:
        out = dict(Total = total_scalars.pop(k))
        for i in sorted(scene_split[k]):
            out[i] = scene_split[k][i]
        for i in sorted(target_split[k]):
            out[i] = target_split[k][i]
        result_path = os.path.join(args.exp_dir, k+'_'+args.results_json)
        with open(result_path, "w") as fp:
            json.dump(out, fp, indent=4)

    out = dict(Total = total_scalars.pop('Total'))
    for i in sorted(total_scalars):
        out[i] = total_scalars[i]
    result_path = os.path.join(args.exp_dir, 'Total_'+args.results_json)
    with open(result_path, "w") as fp:
        json.dump(out, fp, indent=4)