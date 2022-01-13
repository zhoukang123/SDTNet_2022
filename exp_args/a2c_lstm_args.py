from .default_args import args

args.update(
    train_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    train_targets = {'kitchen':["Microwave"],},
    test_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    test_targets = {'kitchen':["Microwave"],},
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,
    },
    obs_dict = {
        'fc':'resnet50_fc_new.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 100000,
    total_eval_epi = 100,
    threads = 4,
    exp_name = 'LstmModel',
    optimizer = 'RMSprop',
    model = 'LstmModel',
    agent = 'A2CLstmAgent',
    runner = 'A2CRunner',
    loss_func = 'basic_loss',
    trainer = 'a2c_train',
    optim_args = dict(lr = 0.0001, alpha = 0.99, eps = 0.1),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 100000,
    nsteps = 10,
    verbose = False,
    gpu_ids = [0],
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
)
model_args_dict = {'action_sz' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)

