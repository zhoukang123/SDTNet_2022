import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_layer import GraphAttentionLayer, SpGraphAttentionLayer
import h5py
import numpy as np
import scipy.sparse as sp
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        A_raw = torch.load ("D:/vnenv-master (2)/vnenv-master (2)/thordata/gcn/obj_direct.dat")
        A = normalize_adj (A_raw).tocsr ().toarray ()
        self.A = torch.nn.Parameter (torch.Tensor (A))
        # 新的有向图，有向图的glove以及有向图的关系权重邻接矩阵
        # objects = open("../thordata/gcn/objects_ori.txt").readlines()
        objects = open ("D:/vnenv-master (2)/vnenv-master (2)/thordata/gcn/direction_obj.txt").readlines ()
        objects = [o.strip () for o in objects]
        self.n = len (objects)
        self.register_buffer ('all_glove', torch.zeros (self.n, 300))
        # glove需要自己构建新的有向图glove,在原有基础上追加新的目标
        # glove = h5py.File("D:/vnenv-master (2)/vnenv-master (2)/thordata/word_embedding/glove_map300d_direct.hdf5","r",)
        glove = h5py.File ("D:/vnenv-master (2)/vnenv-master (2)/thordata/word_embedding/glove_map300d_direct.hdf5",
                           "r", )
        for i in range (self.n):
	        self.all_glove [i, :] = torch.from_numpy (glove [objects [i]] [:])

        glove.close ()

        nhid = 1024
        # Convert word embedding to input for gat
        self.word_to_gat = nn.Linear (300, 512)

        # Convert resnet feature to input for gat
        self.resnet_to_gat = nn.Linear (1000, 512)
        # self.resnet_to_gat = nn.Linear (1000, 512)

        # Gat net
        self.gat1 = GraphAttentionLayer (512 + 512, nhid, dropout=0.2, alpha=0.02)
        self.gat2 = GraphAttentionLayer (nhid, nhid, dropout=0.2, alpha=0.02)
        self.gat3 = GraphAttentionLayer (nhid, 1, dropout=0.2, alpha=0.02)

        self.mapping = nn.Linear (self.n, 512)  # 92到512

    def gat_embed (self, x, params):
	    if params == None:
		    resnet_embed = self.resnet_to_gat (x)
		    word_embedding = self.word_to_gat (self.all_glove)
		    n_steps = resnet_embed.shape [0]
		    resnet_embed = resnet_embed.repeat (self.n, 1, 1)
		    output = torch.cat (
			    (resnet_embed.permute (1, 0, 2), word_embedding.repeat (n_steps, 1, 1)),
			    dim=2
		    )
	    else:
		    resnet_embed = F.linear (
			    x,
			    weight=params ["resnet_to_gcn.weight"],
			    bias=params ["resnet_to_gcn.bias"],
		    )
		    word_embedding = F.linear (
			    self.all_glove,
			    weight=params ["word_to_gcn.weight"],
			    bias=params ["word_to_gcn.bias"],
		    )
		    n_steps = resnet_embed.shape [0]
		    resnet_embed = resnet_embed.repeat (self.n, 1, 1)
		    output = torch.cat (
			    (resnet_embed.permute (1, 0, 2), word_embedding.repeat (n_steps, 1, 1)),
			    dim=2
		    )
	    print ('output', output.size ())  # (512,92,1024)
	    return output

    def forward (self, x, params=None):
	
	    # x = (current_obs)
	    # Convert input to gcn input
	    print ('x,params', x.size ())
	    x = self.gat_embed (x, params)  # (512,1000)
	    print ('x==', x.size ())  # 512,92,1024
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

if __name__ == "__main__":
	model = GAT (3,3,92,0.2,0.2,8)
	input_ = {
		'res18fm': torch.randn (4, 512, 7, 7),
		'score': torch.randn (4, 1000),
		'action_probs': torch.randn (4, 3),
		'hidden': (torch.randn (4, 512), torch.randn (4, 512)),
		'glove': torch.randn (4, 300)
	}
	
	cc = {}
	
	for name, param in model.named_parameters ():
		# Clone and detach.
		param_copied = param.clone ().detach ().requires_grad_ (True)
		cc [name] = param_copied
		
	out = model.forward (input_)
	print (out ['value'])
	# out['value'].mean().backward()
	# out = model.forward(input_)
	# print(out['value'])
	out = model.forward (input_, cc)
	print (out ['value'])