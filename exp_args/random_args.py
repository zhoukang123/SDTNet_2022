from .default_args import args

args.update(
    test_scenes = {
        'kitchen':'21-30',
        'living_room':'21,24-30',
        'bedroom':'21-30',
        'bathroom':'21-30',
    },
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,#Done动作一定必须绑定为None
        #包含Done字符串时，需要智能体自主提出结束episode，不包含时环境会自动判定是否结束
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 40000,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'Random',
    optimizer = 'Adam',
    model = 'DemoModel',
    agent = 'RandomAgent',
    runner = 'A2CRunner',
    loss_func = 'basic_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = args.lr,),
    print_freq = 1000,
    max_epi_length = 200,
    model_save_freq = 40000,
    shuffle = False,
    nsteps = 10,
    verbose = False,
    gpu_ids = -1,
    results_json = "result.json"
)
model_args_dict = {'action_size' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)
