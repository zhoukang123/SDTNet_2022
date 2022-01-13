import math
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter

#有几个问题
#resnet50直接从图像跑，太慢了
#稀疏矩阵乘法的速度
#target的全连接层是300x300
#g

class ScenePriorsGATModel(nn.Module):
    """Scene Priors implementation"""
    def __init__(
        self,
        action_sz,
        state_sz = 8192,
        target_sz = 300,
        ):
        super(ScenePriorsGATModel, self).__init__()
        target_embed_sz = 300
        self.fc_target = nn.Linear(target_sz, target_embed_sz)
        # Observation layer
        self.fc_state = nn.Linear(state_sz, 512)

        # GAT layer
        self.gat = GAT()

        # Merge word_embedding(300) + observation(512) + gcn(512)
       
        self.navi_net = nn.Linear(
            target_embed_sz+ 640, 512)#300+640
        self.navi_hid = nn.Linear(512,512)
        #output
        self.actor_linear = nn.Linear(512, action_sz)
        self.critic_linear = nn.Linear(512, 512)

    def forward(self, model_input):

        x = model_input['fc|4'].reshape(-1, 8192)
    
        y = model_input['glove']
        z = model_input['score']
        # z = model_input ['score']#512,1,1,1000
        print('z',z.size())

        #x = x.view(-1)
        #print(x.shape)
        x = self.fc_state(x)
        x = F.relu(x, True)
        #y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)
        
        #print(self.gat(y))
        z = self.gat(z)#
        
        print('xxxx=',x.size())
        print ('yyyy', y.size ())
        z=z.reshape(4,128)
        print ('zzz=', z.size ())
        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat((x, y, z), dim = 1)
        print('xyzxyzxyz=',xyz.size())#(4,940)
        #xyz = torch.cat ((x, y), dim=1)
        xyz = self.navi_net(xyz)#(4,512)
        print('xyzxyz_nabi_net',xyz.size())
        xyz = F.relu(xyz, True)#(4,512)
        xyz = F.relu(self.navi_hid(xyz), True)#(512,512)
        print('xyz_navi_hid',xyz.size())#(4,512)
        return dict(
            policy=self.actor_linear(xyz),
            value=self.critic_linear(xyz)
            )

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        # Load adj matrix for GAT有向图的邻接矩阵
        #A_raw = torch.load("../thordata/gcn/obj.dat")
        A_raw = torch.load ("D:/vnenv-master (2)/vnenv-master (2)/thordata/gcn/obj_direct.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))
        print('self.A',self.A.size())
		#新的有向图，有向图的glove以及有向图的关系权重邻接矩阵
        #objects = open("../thordata/gcn/objects_ori.txt").readlines()
        objects = open ("D:/vnenv-master (2)/vnenv-master (2)/thordata/gcn/direction_obj.txt").readlines ()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        self.register_buffer('all_glove', torch.zeros(self.n, 300))
		#glove需要自己构建新的有向图glove,在原有基础上追加新的目标
        #glove = h5py.File("D:/vnenv-master (2)/vnenv-master (2)/thordata/word_embedding/glove_map300d_direct.hdf5","r",)
        glove = h5py.File ("D:/vnenv-master (2)/vnenv-master (2)/thordata/word_embedding/glove_map300d_direct.hdf5", "r", )
        for i in range(self.n):
            self.all_glove[i, :] = torch.from_numpy(glove[objects[i]][:])

        glove.close()

        nhid = 1024
        # Convert word embedding to input for gat
        self.word_to_gat = nn.Linear(300, 512)

        # Convert resnet feature to input for gat
        self.resnet_to_gat = nn.Linear(1000, 512)
        #self.resnet_to_gat = nn.Linear (1000, 512)

        # Gat net
        self.gat1 = GraphAttentionConvolution( 512+512, nhid,dropout=0.2,alpha=0.02)#1024,1024
        self.gat2 = GraphAttentionConvolution(nhid, nhid,dropout=0.2,alpha=0.02)#1024,1024
        self.gat3 = GraphAttentionConvolution(nhid, 1,dropout=0.2,alpha=0.02)#1024,1

        self.mapping = nn.Linear(self.n, 512)#92到512

    def gat_embed(self, x, params):
        print('x.xxxx',x.size())#(512,1000)
        if params == None:
            resnet_embed = self.resnet_to_gat(x)
            print('resFeatures=',resnet_embed.size())#512,1,1,512
            word_embedding = self.word_to_gat(self.all_glove)
            print ('wordFeatures=', word_embedding.size ())#92,512
            n_steps = resnet_embed.shape[0]
            #resnet_embed = resnet_embed.repeat(self.n,1)
            #output = torch.cat((resnet_embed,word_embedding))
            #print('output_size=',output.size())#512+92,512
            #print('(resnet_embed.permute (1, 0, 2)=',(resnet_embed.permute (1, 0, 2)).size())
            #print('word_embedding.repeat (n_steps, 1, 1)',word_embedding.repeat (n_steps, 1, 1).size())
            #output = torch.cat((resnet_embed.permute (1, 0, 2), word_embedding.repeat (n_steps, 1, 1)), dim = 2)
            print('n_steps',n_steps)
            print('self.n=',self.n)#92
            resnet_embed=resnet_embed.repeat(self.n,1,1,1)
            # word_embedding=word_embedding.repeat(n_steps,1,1)
            print('resnet_embed=',resnet_embed.size())#92,512,1,512
            print ('word_embedding=', word_embedding.size ())
            resnet_embed=resnet_embed.reshape(92,512,512)
            word_embedding=word_embedding.reshape(92,512,1)
            # word_embedding=word_embedding.reshape(92,512)
            #resnet_embed=resnet_embed.expand_as(word_embedding)
            #word_embedding=word_embedding.expand_as(resnet_embed)
            output = torch.cat ((resnet_embed, word_embedding.repeat(n_steps,1,1)),dim=2)
            
            #output=output.reshape(92,1024)#92*513
            print ('output_size=', output.size ())#
        else:
            resnet_embed = F.linear(
                x,
                weight=params["resnet_to_gcn.weight"],
                bias=params["resnet_to_gcn.bias"],
            )
            word_embedding = F.linear(
                self.all_glove,
                weight=params["word_to_gcn.weight"],
                bias=params["word_to_gcn.bias"],
            )
            n_steps = resnet_embed.shape[0]
            resnet_embed = resnet_embed.repeat(self.n,1,1)
            output = torch.cat(
                (resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)),
                dim=2
                )
        print('output',output.size())#(512,92,1024)
        return output
    def forward(self, x, params = None):

        # x = (current_obs)
        # Convert input to gcn input
        print('x,params',x.size())#512,1,1,1000
        x = self.gat_embed(x, params)#(512,1000)
        print('x==',x.size())#93,512
        print('self_A=',self.A.size())#(92,92)
        if params == None:
            x = F.relu(self.gat1(x, self.A))#512,1024
            print('xgat1=',x.size())
            x = F.relu(self.gat2(x, self.A))#1024,1024
            x = F.relu(self.gat3(x, self.A))#1024,1
            x.squeeze_(-1)
            x = self.mapping(x)
            print('xxx',x.size())
        else:
            gat_p = [
                dict(
                    weight = params[f'gcn{x}.weight'], bias = params[f'gcn{x}.bias']
                    )
                for x in [1,2,3]
                ]
            x = F.relu(self.gat1(x, self.A, gat_p[0]))
            x = F.relu(self.gat2(x, self.A, gat_p[1]))
            x = F.relu(self.gat3(x, self.A, gat_p[2]))
            x.squeeze_(-1)
            x = F.linear(
                    x,
                    weight=params["mapping.weight"],
                    bias=params["mapping.bias"],
                )

        return x
class GraphAttentionConvolution(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features,dropout,alpha,concat=True, bias=True):
        super(GraphAttentionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.droupout=dropout
        self.alpha=alpha
        self.concat=concat
        self.leakyrelu = nn.LeakyReLU (self.alpha)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        print('w:=',self.W.size())
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        
        nn.init.xavier_uniform_(self.W.data,gain=1.414)

        #attention初始权重
        self.A = nn.Parameter (torch.zeros (size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.A.data, gain=1.414)
        self.leakyrelu=nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        
	    #将上次训练的参数保存到weight和bias中
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj, params = None):
        print('input',input.size())#([512,92,1024])
        print('self.W:',self.W.size())#([1024,1024])
        #input=input.reshape((512*92),1024)
        #input=input.permute(1,0,2).permute(0,2,1)[:,:,-1]#92,1024,512直接使用2D
        #print ('input==', input.size ())  #93,512
        #Wh = torch.mm(input, self.W)
        Wh=torch.matmul(input,self.W)
        #print('h=',Wh.size())
        N = Wh.size()[0]
        #print('N=',N)
        a_input=self._prepare_attentional_mechanism_input(Wh)
        #print('a_input',a_input.size())
        #print ('self.A', self.A.size ())
        e=self.leakyrelu(torch.matmul(a_input,self.A).squeeze(2))
        #a_input = torch.cat ([h.repeat (1, N).view (N * N, -1), h.repeat (N, 1)], dim=1).view (N, -1,2 * self.out_features)
        #e = self.leakyrelu (torch.matmul (a_input, self.A).squeeze (2))
        #print('e=',e.size())#92,92
        zero_vec = -9e15 * torch.ones_like (e)
        #print('zero_vec=',zero_vec.size())
        attention = torch.where (adj > 0, e, zero_vec)#把大于0的都变成0
        attention = F.softmax (attention, dim=1)
        #self.droupout=0.2
        attention = F.dropout (attention,self.droupout,training=self.training)
        output = torch.matmul (attention, Wh)
        #print('output=',output.size())
        if params == None:
            #support = torch.matmul(input, self.weight)
            #output = torch.matmul(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output
        elif params!=None:
            print('weight_size',params['weight'].size())
            support = torch.matmul(input, params['weight'])
            output = torch.matmul(attention, support)
            if self.bias is not None:
                return output + params['bias']
        elif self.concat:
            return F.elu(output)
        else:
            return output

    def _prepare_attentional_mechanism_input (self, Wh):
        N = Wh.size () [0]  # number of nodes
    
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
    
        Wh_repeated_in_chunks = Wh.repeat_interleave (N, dim=0)
        Wh_repeated_alternating = Wh.repeat (N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)
    
        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN
    
        all_combinations_matrix = torch.cat ([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
    
        return all_combinations_matrix.view (N, N, 2 * self.out_features)
    
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

if __name__ == "__main__":
    model = ScenePriorsGATModel(9)
    input1 = torch.randn(4,8192)#resent
    input2 = torch.randn(4,300)#glve
    input3 = torch.randn(1,512,1,1000)
    #input3= torch.randn (512, 1,1,1000)#score
    #out = model.forward({'fc|4':input1, 'glove':input2, 'RGB':input3})
    out = model.forward ({'fc|4': input1, 'glove': input2, 'score': input3})
    print(out['policy'])
    print(out['value'])

