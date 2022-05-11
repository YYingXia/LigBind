import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import knn_graph,radius_graph
from torch_geometric.nn import global_max_pool,max_pool,avg_pool,global_add_pool, global_mean_pool


class EdgeModel(torch.nn.Module):
    def __init__(self,x_ind,edge_ind,u_ind,hs,dropratio,bias=True):
        super(EdgeModel, self).__init__()
        self.u_ind = u_ind
        self.mlp = nn.Sequential(
                              nn.Conv1d(x_ind*2+edge_ind+u_ind, hs,kernel_size=1,bias=bias),
                              nn.BatchNorm1d(hs),
                              nn.ReLU(inplace=True),
                              nn.Dropout(dropratio),
                              nn.Conv1d(hs,hs,kernel_size=1,bias=bias),
                              nn.BatchNorm1d(hs))

    def forward(self, src, dst, edge_attr,u,batch):
        out = [src,dst,edge_attr]
        if u is not None and self.u_ind!=0:
            out.append(u[batch])
        out = torch.cat(out,dim=-1)
        out = out.permute(1,0).unsqueeze(0)
        out = self.mlp(out).squeeze().permute(1,0)
        return out

class NodeModel(torch.nn.Module):  # neighbor node无权重
    def __init__(self,x_ind,edge_ind,u_ind,hs,dropratio,aggr,bias=True,edge_attr_num=1):
        super(NodeModel, self).__init__()
        self.aggr = aggr
        self.u_ind = u_ind
        if edge_ind != 0:
            self.mlp1 = nn.Sequential(
                                  nn.Conv1d(edge_attr_num*edge_ind, hs,kernel_size=1,bias=bias),
                                  nn.BatchNorm1d(hs),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropratio),
                                  nn.Conv1d(hs,hs,kernel_size=1,bias=bias),
                                  nn.BatchNorm1d(hs))

        self.mlp2 = nn.Sequential(
            nn.Conv1d(x_ind + edge_ind + u_ind, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(hs, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs))

    def forward(self, x, edge_index, edge_attr,u, batch):
        out = [x]
        if u is not None and self.u_ind!=0:
            out.append(u[batch])

        row, col = edge_index
        if edge_attr is not None:
            if self.aggr == 'max':
                out.append(global_max_pool(edge_attr, col, x.size(0)))
            elif self.aggr == 'add':
                out.append(global_add_pool(edge_attr, col, x.size(0)))
            elif self.aggr == 'mean':
                out.append(global_mean_pool(edge_attr, col, x.size(0)))
        else:
            if self.aggr == 'max':
                out.append(global_max_pool(x[row], col, x.size(0)))
            elif self.aggr == 'add':
                out.append(global_add_pool(x[row], col, x.size(0)))
            elif self.aggr == 'mean':
                out.append(global_mean_pool(x[row], col, x.size(0)))

        out = torch.cat(out,dim=1).permute(1,0).unsqueeze(0)
        out = self.mlp2(out).squeeze().permute(1,0)
        return out

class GlobalModel(torch.nn.Module):  # neighbor node无权重
    def __init__(self,x_ind,u_ind,hs,dropratio,aggr,bias=True,poolratio=1):
        super(GlobalModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(x_ind + u_ind, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(hs, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs))

        self.hs = hs
        self.aggr = aggr
        self.poolratio = poolratio

    def forward(self, x,u, batch, polar_pos=None):
        out = []
        if u is not None:
            out.append(u)

        if self.aggr == 'max':
            out.append(global_max_pool(x, batch))
        elif self.aggr == 'add':
            out.append(global_add_pool(x, batch))
        elif self.aggr == 'mean':
            out.append(global_mean_pool(x, batch))

        out = torch.cat(out, dim=1).unsqueeze(-1)
        out = self.mlp(out).squeeze(-1) # batch * hs
        return out

class MetaEncoder(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaEncoder, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        if edge_attr is not None:
            edge_attr = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        u = self.global_model(x, u, batch,polar_pos)

        return x, edge_attr, u

class MetaGRU(torch.nn.Module):
    def __init__(self, gru_steps,e_hs,x_hs,u_hs,bias=True,edge_model=None, node_model=None, global_model=None):
        super(MetaGRU, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        if edge_model is not None:
            self.edge_rnn = torch.nn.GRUCell(e_hs,e_hs,bias=bias)
        self.node_rnn = torch.nn.GRUCell(x_hs,x_hs,bias=bias)
        self.global_rnn = torch.nn.GRUCell(u_hs,u_hs,bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        x_cat_out = [x]
        global_out = [u]
        for i in range(self.gru_steps):
            if edge_attr is not None:
                edge_attr_out = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
                edge_attr = self.edge_rnn(edge_attr_out,edge_attr)

            x_out = self.node_model(x, edge_index, edge_attr, u, batch)
            x = self.node_rnn(x_out,x)
            x_cat_out.append(x)

            u_out = self.global_model(x, u, batch,polar_pos)
            u = self.global_rnn(u_out,u)
            global_out.append(u)

        x_cat_out = torch.cat(x_cat_out,dim=1)
        global_out = torch.cat(global_out,dim=1)
        return x_cat_out, global_out

class MetaMLP(torch.nn.Module):
    def __init__(self, gru_steps,edge_model=None, node_model=None, global_model=None):
        super(MetaMLP, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        x_cat_out = [x]
        global_out = [u]
        for i in range(self.gru_steps):
            edge_attr = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
            x = self.node_model(x, edge_index, edge_attr, u, batch)
            u = self.global_model(x, u, batch,polar_pos)
            x_cat_out.append(x)
            global_out.append(u)

        x_cat_out = torch.cat(x_cat_out,dim=1)
        global_out = torch.cat(global_out,dim=1)
        return x_cat_out, global_out

class MetaBind_MultiEdges(torch.nn.Module):
    def __init__(self, fea_ext_fixed, gru_steps,x_ind,edge_ind,x_hs,e_hs,u_hs,dropratio,bias,edge_method,r,aggr,dist,max_nn,
                 stack_method,apply_edgeattr,apply_nodeposemb):
        super(MetaBind_MultiEdges, self).__init__()
        self.dist = dist
        self.max_nn = max_nn
        self.u_ind = 0
        self.edge_method = edge_method
        if apply_nodeposemb is False:
            x_ind -= 1
        self.x_bn = nn.BatchNorm1d(x_ind)
        # self.x_bn = nn.BatchNorm1d(44)
        self.edge_bn = nn.BatchNorm1d(2)
        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)  # 欧氏距离
        self.cossim = nn.CosineSimilarity(dim=1)
        self.r = r
        self.apply_edgeattr = apply_edgeattr
        self.apply_nodeposemb = apply_nodeposemb

        self.aggr = aggr

        self.encoder = MetaEncoder(edge_model=EdgeModel(x_ind=x_ind,edge_ind=edge_ind,u_ind=0,hs=e_hs,dropratio=dropratio,bias=bias),
                                   node_model=NodeModel(x_ind=x_ind,edge_ind=e_hs,u_ind=0,hs=x_hs,dropratio=dropratio,bias=bias,aggr=aggr,edge_attr_num=1),
                                   global_model=GlobalModel(x_ind=x_hs,u_ind=0,hs=u_hs,dropratio=dropratio,bias=bias,aggr=aggr))
        if stack_method == 'GRU':
            self.stacked_GN = MetaGRU(gru_steps=gru_steps,e_hs=e_hs,x_hs=x_hs,u_hs=u_hs,bias=True,
                                  edge_model=EdgeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=0,hs=e_hs,dropratio=dropratio,bias=bias),
                                  node_model=NodeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=0,hs=x_hs,dropratio=dropratio,bias=bias,aggr=aggr,edge_attr_num=1),
                                  global_model=GlobalModel(x_ind=x_hs,u_ind=u_hs,hs=u_hs,dropratio=dropratio,bias=bias,aggr=aggr))
        elif stack_method == 'MLP':
            self.stacked_GN = MetaMLP(gru_steps=gru_steps,
                                      edge_model=EdgeModel(x_ind=x_hs, edge_ind=e_hs, u_ind=0, hs=e_hs,dropratio=dropratio, bias=bias),
                                      node_model=NodeModel(x_ind=x_hs, edge_ind=e_hs, u_ind=0, hs=x_hs,dropratio=dropratio, bias=bias, aggr=aggr, edge_attr_num=1),
                                      global_model=GlobalModel(x_ind=x_hs, u_ind=u_hs, hs=u_hs, dropratio=dropratio,bias=bias, aggr=aggr))

        emb_dim = u_hs * (gru_steps + 1)

        fea_ext_fixed_list = []
        if fea_ext_fixed:
            for i, p in enumerate(self.parameters()):
                p.requires_grad = False
                fea_ext_fixed_list.append(i)
        # print('fixed params:', fea_ext_fixed_list)

        self.score = nn.Sequential(
            nn.Linear(emb_dim,u_hs, bias=False),
            nn.BatchNorm1d(u_hs),
            nn.ReLU(inplace=True),
            nn.Linear(u_hs,1))

    # def forward(self, *argv):
    def forward(self, data):

        # if len(argv) == 4:
        #     x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        # elif len(argv) == 1:
        #     data = argv[0]
        x, pos, batch = data.x, data.pos, data.batch
        edge_index = radius_graph(pos, r=self.r, batch=batch, loop=True, max_num_neighbors=self.max_nn)
        edge_attr = torch.cat([self.pdist(pos[edge_index[0]], pos[edge_index[1]]) / self.r,
                                 (self.cossim(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1) + 1) / 2],
                                dim=1)

        x = torch.cat([x, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist],dim=-1)  # 加入node离原点的距离作为特征


        # else:
        #     raise ValueError("unmatched number of arguments.")

        edge_attr = self.edge_bn(edge_attr.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        x = self.x_bn(x.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)  # float16会导致batchnorm输出为nan/inf
        x, edge_attr, u = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)
        x_output, global_output = self.stacked_GN(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        # logit = self.score(global_output).squeeze(1)
        embs = self.score[0](global_output)
        logit = self.score[1:](embs).squeeze(1)

        return logit, global_output, x_output

    def forward_emb(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        edge_index = radius_graph(pos, r=self.r, batch=batch, loop=True, max_num_neighbors=self.max_nn)
        edge_attr = torch.cat([self.pdist(pos[edge_index[0]], pos[edge_index[1]]) / self.r,
                               (self.cossim(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1) + 1) / 2],
                              dim=1)

        x = torch.cat([x, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist],
                      dim=-1)  # 加入node离原点的距离作为特征
        edge_attr = self.edge_bn(edge_attr.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        x = self.x_bn(x.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)  # float16会导致batchnorm输出为nan/inf
        x, edge_attr, u = self.encoder(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)
        x_output, global_output = self.stacked_GN(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)

        embs = self.score[0](global_output)
        logit = self.score[1:](embs).squeeze(1)

        return logit, embs

class Pair_model_spe(torch.nn.Module):
    def __init__(self, protein_model, prot_emb_dim, mol_emb_dim, emb_dim, n_cluster):
        super(Pair_model_spe, self).__init__()

        self.ligand_cluster = n_cluster
        self.protein_model = protein_model
        self.mol_model = nn.Sequential(
            nn.Linear(mol_emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim))

        self.weights_mlp = nn.Sequential(
            nn.Linear(prot_emb_dim + emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, self.ligand_cluster))

        self.weights_lin1 = nn.Linear(prot_emb_dim + emb_dim, self.ligand_cluster, bias=True)
        self.weights_lin2 = nn.Linear(prot_emb_dim + emb_dim, self.ligand_cluster, bias=False)

        self.clf = nn.Sequential(
            nn.Linear(prot_emb_dim + emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, self.ligand_cluster))

    def forward(self, prot_data, mol_emb, apply_weight=True):
        _, prot_emb, res_emb = self.protein_model(prot_data)
        if len(mol_emb.shape) == 1:
            mol_emb = mol_emb.unsqueeze(0)
            mol_emb = mol_emb.repeat(prot_emb.shape[0], 1)
        mol_emb = self.mol_model(mol_emb)
        pair_emb = torch.cat([prot_emb, mol_emb], dim=1)
        logit = self.clf(pair_emb)
        if apply_weight:
            weights = self.weights_mlp(pair_emb)
            weights = F.softmax(weights)
            logit = weights * logit
            logit = torch.sum(logit, dim=1)
        else:
            logit = torch.mean(logit, dim=1)
        return logit

class Pair_model_gen(torch.nn.Module):
    def __init__(self, protein_model, prot_emb_dim, mol_emb_dim, emb_dim):
        super(Pair_model_gen, self).__init__()
        self.protein_model = protein_model
        self.mol_model = nn.Sequential(
            nn.Linear(mol_emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim,emb_dim))

        self.clf = nn.Sequential(
            nn.Linear(prot_emb_dim+emb_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim,1))
    def forward(self, prot_data, mol_emb):
        _, prot_emb, res_emb = self.protein_model(prot_data)
        if len(mol_emb.shape) == 1:
            mol_emb = mol_emb.unsqueeze(0)
            mol_emb = mol_emb.repeat(prot_emb.shape[0],1)
        mol_emb = self.mol_model(mol_emb)

        pair_emb = torch.cat([prot_emb, mol_emb], dim=1)
        pair_logit = self.clf(pair_emb).squeeze(1)
        return pair_logit