from .default_args import args

args.update(
    #train_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    #train_targets = {'kitchen':["Microwave"],},
    #test_scenes = {'kitchen':'25',},#{'bathroom':[31],},
    test_scenes = {
        'kitchen':'21-30',
        #'living_room':'21-22,24-30',
        #'bedroom':'21-30',
        #'bathroom':'21-30',
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
        'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        #'Done':None,
        #包含Done字符串时，需要智能体自主提出结束episode，不包含时环境会自动判定是否结束
    },
    obs_dict = {
        'res18fm':'resnet18_featuremap.hdf5',
         #'score':'resnet50_score.hdf5'
        },
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 1e8,
    total_eval_epi = 250,
    threads = 1,
    exp_name = 'a3cgcnsavn',
    optimizer = 'SharedRMSprop',
    model = 'GCNSAVN',
    agent = 'OriSavnAgent',
    runner = 'SavnRunner',
    tester = 'savn_test',
    loss_func = 'savn_loss',
    trainer = 'ori_savn_train',
    optim_args = dict(lr = 0.0001),
    inner_lr = 0.0001,
    print_freq = 100000,
    max_epi_length = 200,
    model_save_freq = 1e7,
    nsteps = 6,
    verbose = False,
    gpu_ids = -1,
    shuffle = False,
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result.json"
)
model_args_dict = {'action_sz' : len(args.action_dict),'nsteps':args.nsteps}
args.update(
    model_args = model_args_dict,
)

