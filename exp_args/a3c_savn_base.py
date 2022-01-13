from .default_args import args

args.update(
    train_scenes = {
        'kitchen':'25',
        },#{'bathroom':[31],},
    train_targets = {'kitchen':["Microwave", 'Towel'],},
    test_scenes = {
        'kitchen':'21-30',
        'living_room':'21-30',
        #'bedroom':'22,24,26,28,30',
        #'bathroom':'22,24,26,28,30',
    },
    test_targets = {
        'kitchen':[
            "Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",
            ],
        'living_room':[
            "Pillow", "Laptop", "Television",
            "GarbageCan", "Box", "Bowl",
            ],
        #'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        #'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    test_sche_dir = '../thordata/test_schedule',
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,
        #'Done':None,#Done动作一定必须绑定为None
        #包含Done字符串时，需要智能体自主提出结束episode，不包含时环境会自动判定是否结束
    },
    obs_dict = {
        'res18fm':'resnet18_featuremap.hdf5',
        #'fc':'resnet50_fc.hdf5',
        #'score':'resnet50_score.hdf5'
        },
    target_dict = {
        #'glove':'../thordata/word_embedding/word_embedding.hdf5',
        'glove':'../thordata/word_embedding/word_embedding.hdf5'
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 100000,
    total_eval_epi = 100,
    threads = 2,
    exp_name = 'savnBaseModel',
    optimizer = 'RMSprop',
    model = 'BaseModel',
    agent = 'A3CLstmAgent',
    runner = 'A3CRunner',
    loss_func = 'basic_loss',
    trainer = 'a3c_train',
    optim_args = dict(lr = 0.0001, alpha = 0.99, eps = 0.1),
    print_freq = 1000,
    max_epi_length = 200,
    model_save_freq = 25000,
    nsteps = 40,
    verbose = False,
    gpu_ids = [0],
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result_savnbase.json"
)
model_args_dict = {'action_sz' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)

