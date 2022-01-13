from .default_args import args

args.update(
    
    test_scenes = {
        'kitchen':'21-30',
        #'living_room':'21-30',
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
    total_train_frames = 40000,
    total_eval_epi = 250,
    threads = 1,
    exp_name = 'savn',
    optimizer = 'SharedAdam',
    model = 'SAVN',
    agent = 'OriSavnAgent',
    runner = 'SavnRunner',
    loss_func = 'savn_loss',
    trainer = 'ori_savn_train',
    tester = 'savn_test',
    optim_args = dict(lr = 0.0001),
    inner_lr = 0.0001,
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 40000,
    nsteps = 6,
    verbose = False,
    gpu_ids = [0],
    shuffle = False,
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result.json"
)
model_args_dict = {'action_sz' : len(args.action_dict),'nsteps':args.nsteps}
args.update(
    model_args = model_args_dict,
)

