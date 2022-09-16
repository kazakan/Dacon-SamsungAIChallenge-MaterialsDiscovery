import torch.nn as nn
import torch_geometric.nn as gnn
import torch

from .MiddleModule import MiddleModule1

class ResidualGConv(nn.Module):

    def __init__(self,
            n_node_in_dims : int,
        ):
        super().__init__()

        self.gconv = gnn.GCNConv(n_node_in_dims,n_node_in_dims)
        self.gelu = nn.GELU()
        #self.dropout = nn.Dropout(p=0.35)

    def forward(self,x,edge_index):
        #res = self.dropout(x)
        res = self.gconv(x,edge_index)
        res = self.gelu(res)

        return x + res


class GNN1(MiddleModule1):

    def __init__(self,
            n_node_features : int,
            output_dim : int,
        ):
        super().__init__(
            name="GNN1",
            mean_g=0.6548,
            std_g=0.3599,
            mean_ex=0.5892,
            std_ex=0.3365
        )

        # embed atom with its proton number
        self.embed_proton_no = nn.Embedding(119,n_node_features)

        self.res_gconv1 = ResidualGConv(n_node_features)
        self.res_gconv2 = ResidualGConv(n_node_features)
        self.res_gconv3 = ResidualGConv(n_node_features)
        self.res_gconv4 = ResidualGConv(n_node_features)

        self.agg = gnn.MeanAggregation()

        self.after_path = nn.Sequential(
            nn.Linear(n_node_features,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64,output_dim)
        )

        self.mseloss = nn.MSELoss()

        self.double()


    def forward(self,batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.embed_proton_no(x)
        x = self.res_gconv1(x,edge_index)
        x = self.res_gconv2(x,edge_index)
        x = self.res_gconv3(x,edge_index)
        x = self.res_gconv4(x,edge_index)

        x = gnn.global_mean_pool(x,batch.batch)

        x = self.after_path(x)

        return x


class ResidualGATBlock(nn.Module):
    def __init__(self,n_features,edge_dim=1):
        super().__init__()

        self.gat = gnn.GATv2Conv(
            n_features,
            n_features,
            dropout=0.15,
            edge_dim=edge_dim
        )

        self.gelu = nn.GELU()
        self.bnorm = gnn.norm.BatchNorm(n_features)

    def forward(self,x,edge_index,edge_attr):
        x = x + self.gat(x,edge_index,edge_attr = edge_attr)
        x = self.gelu(x)
        x = self.bnorm(x)
        return x


class ABlock(nn.Module):
    def __init__(self,
            n_node_in_dims : int,
            n_gaussian : int = 50,
            n_cf_filters : int = 128,
            r : float = 4.0,
        ):
        super().__init__()
        self.r = r

        self.resgat = ResidualGATBlock(n_node_in_dims)

        self.interaction = gnn.models.schnet.InteractionBlock(
            n_node_in_dims,
            n_gaussian,
            n_cf_filters,
            self.r
        )

        self.gconv = gnn.GCNConv(n_node_in_dims,n_node_in_dims)
        self.gelu = nn.GELU()
        self.bnorm = gnn.norm.BatchNorm(n_node_in_dims)
        self.distance_expansion = gnn.models.schnet.GaussianSmearing(0.0, r, n_gaussian)

    def forward(self,x,edge_index,edge_attr,pos,batch):
        edge_index_pos = gnn.radius_graph(pos, r=self.r, batch=batch,
                                  max_num_neighbors=31)
        row, col = edge_index_pos
        edge_weight_by_pos = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr_by_pos = self.distance_expansion(edge_weight_by_pos)

        x = self.resgat(x,edge_index,edge_attr = edge_attr)

        x = x + self.interaction(x,edge_index_pos,edge_weight_by_pos,edge_attr_by_pos)
        x = self.gelu(x)
        x = self.bnorm(x)
        return x


class GNN2(MiddleModule1):

    def __init__(self,
            n_node_features : int,
            n_preblocks : int = 2,
            n_afterblocks : int = 2
        ):
        super().__init__(
            name="GNN2",
            mean_g=0.6548,
            std_g=0.3599,
            mean_ex=0.5892,
            std_ex=0.3365
        )

        # embedding layer
        self.embed_proton_no = nn.Embedding(119,n_node_features)

        self.pre_g = nn.ModuleList()
        self.pre_ex = nn.ModuleList()
        self.after_blocks = nn.ModuleList()

        for _ in range(n_preblocks):
            self.pre_g.append(ABlock(n_node_features))
            self.pre_ex.append(ABlock(n_node_features))

        for _ in range(n_afterblocks):
            self.after_blocks.append(ResidualGATBlock(n_node_features*2))

        self.agg = gnn.MeanAggregation()

        self.mlp = nn.Sequential(
            nn.Linear(n_node_features*2,128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,2)
        )

        self.double()


    def forward(self,batch):
        x, edge_index = batch.x, batch.edge_index
        pos_g, pos_ex = batch.pos_g, batch.pos_ex

        edge_g_len = batch.edge_attr_g[:,1].double()
        edge_ex_len =  batch.edge_attr_ex[:,1].double()

        x = self.embed_proton_no(x)

        x_g = x.clone()
        x_ex = x.clone()

        for block in self.pre_g:
            x_g = block(x_g,edge_index,edge_g_len,pos_g,batch.batch)

        for block in self.pre_ex:
            x_ex = block(x_ex,edge_index,edge_ex_len,pos_ex,batch.batch)

        x = torch.cat((x_g,x_ex),axis=-1)

        for block in self.after_blocks:
            x = block(x,edge_index,None)
    
        x = gnn.global_mean_pool(x,batch.batch)

        x = self.mlp(x)

        return x


class GNN3(MiddleModule1):

    def __init__(self,
            n_node_features : int,
            n_preblocks : int = 2,
            n_afterblocks : int = 2
        ):
        super().__init__(
            name="GNN3",
            mean_g=0.6548,
            std_g=0.3599,
            mean_ex=0.5892,
            std_ex=0.3365
        )

        # embedding layer
        self.embed_proton_no = nn.Embedding(119,n_node_features)

        self.pre_g = nn.ModuleList()
        self.pre_ex = nn.ModuleList()
        self.after_blocks = nn.ModuleList()

        for _ in range(n_preblocks):
            self.pre_g.append(ABlock(n_node_features))
            self.pre_ex.append(ABlock(n_node_features))

        for _ in range(n_afterblocks):
            self.after_blocks.append(ResidualGATBlock(n_node_features*2,edge_dim=2))

        self.agg = gnn.MeanAggregation()

        self.mlp = nn.Sequential(
            nn.Linear(n_node_features*2,128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,2)
        )

        self.double()


    def forward(self,batch):
        x, edge_index = batch.x, batch.edge_index
        pos_g, pos_ex = batch.pos_g, batch.pos_ex

        edge_g_len = batch.edge_attr_g[:,1].double()
        edge_ex_len =  batch.edge_attr_ex[:,1].double()

        x = self.embed_proton_no(x)

        x_g = x.clone()
        x_ex = x.clone()

        for block in self.pre_g:
            x_g = block(x_g,edge_index,edge_g_len,pos_g,batch.batch)

        for block in self.pre_ex:
            x_ex = block(x_ex,edge_index,edge_ex_len,pos_ex,batch.batch)

        x = torch.cat((x_g,x_ex),axis=-1)

        edge_lens = torch.column_stack((edge_g_len,edge_ex_len))
        for block in self.after_blocks:
            x = block(x,edge_index,edge_lens)
    

        x = gnn.global_mean_pool(x,batch.batch)

        x = self.mlp(x)

        return x
