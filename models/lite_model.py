import torch
import torch.nn as nn
import torch.nn.functional as F
#model里不再分别设置state、target之类的概念，而是要对input中的每一个成分进行处理
#target可能有不同成分，以后修复
class LiteModel(torch.nn.Module):
    """一个简单的模型,model里不再分别设置state、target之类的概念，
    而是要对input中的每一个成分进行处理"""
    def __init__(
        self,
        action_sz,
        state_sz,
        target_sz,
    ):
        super(LiteModel, self).__init__()
        
        target_embed_sz = 512
        state_embed_sz = 512
        self.embed_state = nn.Linear(state_sz, state_embed_sz)
        self.embed_target = nn.Linear(target_sz, target_embed_sz)
        middle_sz = state_embed_sz + target_embed_sz

        #navigation architecture
        navi_arch_out_sz = 512
        self.navi_net = nn.Linear(middle_sz,navi_arch_out_sz)

        #output
        self.actor_linear = nn.Linear(navi_arch_out_sz, action_sz)
        self.critic_linear = nn.Linear(navi_arch_out_sz, 1)

    def forward(self, model_input):
        '''保证输入的数据都是torch的tensor'''
        
        state_embed2 = self.embed_state(model_input['fc'])
        state_embed = F.relu(state_embed2)
        #如果输入的envstate是一个batch，就要repeat word embedding

        target_embed = F.relu(self.embed_target(model_input['glove']))
        x = torch.cat((state_embed, target_embed), dim=1)

        x = F.relu(self.navi_net(x))
        
        return dict(
            policy=self.actor_linear(x),
            value=self.critic_linear(x)
            )

if __name__ == "__main__":
    model = LiteModel(4,2048,300)
    input1 = torch.randn(1,2048)
    input2 = torch.randn(1,300)
    out = model.forward(dict(fc=input1, glove=input2))
    print(out['policy'])
    print(out['value'])
    out = model.forward(dict(fc=input1, glove=input2))
    print(out['policy'])
    print(out['value'])

