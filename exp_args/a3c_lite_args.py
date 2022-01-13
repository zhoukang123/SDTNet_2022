from .default_args import args

args.update(
    train_scenes = {'kitchen':'25'},
    train_targets = {'kitchen':["Microwave", 'Sink'],},
    test_scenes = {'kitchen':'25',},
    test_targets = {'kitchen':["Microwave", 'Sink'],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'Done':None,
    },
    obs_dict = {
        'fc':'resnet50_fc.hdf5'
    },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5'
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 1e6,
    total_eval_epi = 1000,
    threads = 4,
    exp_name = 'A3CLiteDemo',
    optimizer = 'SharedAdam',
    model = 'LiteModel',
    agent = 'A3CAgent',
    runner = 'A3CRunner',
    loss_func = 'basic_loss',
    trainer = 'a3c_train',
    tester = 'a3c_test',
    optim_args = dict(lr = 0.0001,),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 200000,
    nsteps = 10,
    gpu_ids = -1,
)
model_args_dict = dict(
        action_sz = len(args.action_dict),
        state_sz = 2048,
        target_sz = 300,
    )
args.update(
    model_args = model_args_dict,
)
