from utils.vnenv_args import VNENVargs

args_dict = dict(
    verbose = False,#为True时将在控制台打印很多信息，调试用
    visulize = False,
    seed = 1114,#随机数生成种子
    gpu_ids = -1,#指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3] 
    total_train_frames = 6e3,#指定训练多少frames
    max_epi_length = 100,#每个episode的最大步长，即agent在episode中的最大行动数
    total_eval_epi = 1000,#指定测试时测试多少个episode
    print_freq = 1000,#每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq = 10000,#每进行n个episode，就保存一次模型参数
    results_json = 'result.json',#测试完成后结果输出到哪个文件
    load_model_dir = '',#要读取的模型参数的完整路径，包括文件名
    exps_dir = '../EXPS',#保存所有实验文件夹的路径
    exp_name = 'demo_exp',#将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir = '',#单次实验的完整路径，会根据时间自动生成
    test_sche_dir = '',#测试序列的json文件夹
    nsteps = 5,#更新梯度的频率，每n步进行一次loss计算并更新梯度
    threads = 4,#线程数
    offline_data_dir = 'D:/AI2thor_offline_data_2.0.2/',#数据集的位置，该路径下应该是那些FloorPlan开头的文件夹才对
    lr = 0.0002,#learning rate
    shuffle = True,#在为线程随机分配训练集时是否打乱顺序

    #componets args 
    #决定要调用的环境、智能体等等，需要新增时，写到对应的模组下，并在__init__中 import
    #直接写类名或函数名
    trainer = 'a2c_train',
    tester = 'a3c_test',
    loss_func = 'basic_loss',
    model = 'LiteModel',#需要的参数见下
    agent = 'BasicAgent',
    runner = 'A2CRunner',#用于指导环境和智能体交互产生数据的类
    optimizer = 'SGD',#需要的参数见下
    env = 'DiscreteEnvironment',

    #interact args
    #动作参数字典，键值为该动作的字符串，值为一个列表，元素为该数据集支持的最小动作元的list。
    #最小运动元的格式为字母+角度。例如m0的含义为，向0度方向move一小步
    action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
        'LookUp':['p-30'],
        'LookDown':['p30'],
        'Done':None,#Done动作一定必须绑定为None
        #包含Done字符串时，需要智能体自主提出结束episode，不包含时环境会自动判定是否结束
    },
    #目标参数字典，键值为目标的某个表示的类型，一个目标可以有多种表示。值为读取该表示的数据文件所在的路径
    target_dict = {
        'glove':'../thordata/word_embedding/word_embedding.hdf5',
    },
    #状态参数字典，键值为状态的某个表示的类型，一个状态可以有多个表示。
    #值为读取该表示的数据文件的文件名路径，当选定某个scene后，才能找到对应改scene的文件
    obs_dict = {
        #'fc':'resnet50_fc.hdf5',#注意bedroom和bathroom没有fc文件，另外223房间有毛病
        #fc:1x2048x1x1
        },
    #回报参数字典，键值为某种事件的字符串，值为该事件对应的回报
    #目前支持以下四种事件。需要新增事件时，需要再写代码
    reward_dict = {
        'collision':-0.1,
        'step':-0.01,
        'SuccessDone':10,
        'FalseDone':0,
        #'angle_restrict': -0.05
    },

    #data args
    #指定训练用的场景。例如厨房用1-20这20个房间作为训练集，则写入字符串'1-20'
    train_scenes = {
        'kitchen':'1-20',
        'living_room':'1-20',
        'bedroom':'1-20',
        'bathroom':'1-20',
    },
    #指定测试用的场景。
    test_scenes = {
        'kitchen':'21-30',
        'living_room':'21-22,24-30',
        'bedroom':'21-30',
        'bathroom':'21-30',
    },
    #指定训练选择的目标的字符串
    train_targets = {
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
    #指定测试时选择的目标的字符串
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
    grid_size = 0.25,
    rotate_angle = 45,
    move_angle = 45,
    horizon_angle = 30,
    chosen_targets = None,#默认值为None时则无限制,这个表是针对env目前能加载的所有scene而言的
    debug = False,
)
#模型参数，这一块儿目前写得还不够灵活，你的模型需要什么参数，就都写在这里
#然后在初始化模型的时候，直接用**args.model_args传入就行了
model_args_dict = dict(
        action_sz = len(args_dict['action_dict']),
        state_sz = 2048,
        target_sz = 300,
    )
#优化器参数，这一块儿目前写得还不够灵活，你选择的需要什么参数，就都写在这里
#然后在初始化优化器的时候，第一个参数为model的parameters，然后跟着**args.optim_args传入就行了
optim_args = dict(
        lr = args_dict['lr'],
        ),

args_dict.update(model_args = model_args_dict, optim_args = optim_args)

args = VNENVargs(args_dict)
