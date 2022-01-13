import json

class VNENVargs:
    def __init__(self, args_dict = None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        self.update(args_dict, **kwargs)

    def update(self, args_dict = None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        if args_dict != None:
            for k in args_dict:
                setattr(self, k, args_dict[k])
        if kwargs !=None:
            for k in kwargs:
                setattr(self, k, kwargs[k])

    def save_args(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def args_pack(self):
        aa = dict(
            env_args = dict(
                offline_data_dir = self.offline_data_dir,
                action_dict = self.action_dict,
                target_dict = self.target_dict,
                obs_dict = self.obs_dict,
                reward_dict = self.reward_dict,
                max_steps = self.max_epi_length,
                grid_size = self.grid_size,
                rotate_angle = self.rotate_angle,
            ),
            agent_args = dict(),
            runner_args = dict(),
            model_args = dict(),
            optim_args = dict()

        )
        self.update(aa)

if __name__ == "__main__":
    args = VNENVargs(a = 1, s= 2, vsvv = 3)
    args.save_args('./test.json')



        

