from .default_args import args

args.update(
    train_scenes = {'kitchen':'10-30',},#{'bathroom':[31],},
    train_targets = {'kitchen':["Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",],
                     },
    test_scenes = {
        'kitchen':'25',
        #'living_room':'21-22,24-30',
        #'bedroom':'21-30',
        #'bathroom':'21-30',
    },
    test_targets = {
         # 'kitchen':[
         #     "Toaster", "Microwave", "Fridge",
         #    "CoffeeMaker", "GarbageCan", "Box", "Bowl",
         #     ],
            'kitchen':[
              "Microwave",
              ],
        #'living_room':[
            #"Pillow", "Laptop", "Television",
            #"GarbageCan", "Box", "Bowl",
            #],
        #'bedroom':["HousePlant", "Lamp", "Book", "AlarmClock"],
        #'bathroom':["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
    },
    #test_sche_dir = '../thordata/test_schedule',
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
        'score':'resnet50_score.hdf5'
        },
    target_dict = {
        'glove':'E:/vnenv-master (2)/thordata/word_embedding/word_embedding_new.hdf5',
    },
    grid_size = 0.25,
    rotate_angle = 45,
    total_train_frames = 1000000,#100万
    total_eval_epi = 250,
    threads = 1,
    exp_name = 'GcnBaseModel',
    optimizer = 'RMSprop',
    model = 'GcnBaseModel',
    agent = 'SavnAgent',#SavnAgent
    runner = 'A3CNoMaskRunner',
    loss_func = 'basic_loss_no_mask',
    trainer = 'a3c_train',
    optim_args = dict(lr = 0.0001, alpha = 0.99, eps = 0.1),
    print_freq = 1000,
    max_epi_length = 100,
    model_save_freq = 40000,
    nsteps = 40,
    verbose = False,
    gpu_ids = -1,
    #load_model_dir = '../check_points/A2CDemoModel_40000_2020-05-20_10-49-28.dat',
    results_json = "result_gcnbase.json"
)
model_args_dict = {'action_sz' : len(args.action_dict)}
args.update(
    model_args = model_args_dict,
)