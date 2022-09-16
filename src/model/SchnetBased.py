import torch.nn as nn
import torch_geometric.nn as gnn
import torch

from .MiddleModule import MiddleModule1


class SchBased1(MiddleModule1):

    def __init__(self,
            mean_g=0.6548,
            std_g=0.3599,
            mean_ex=0.5892,
            std_ex=0.3365
        ):
        super().__init__(name="SchBased1")

        # embed atom with its proton number

        self.sch_ex_1 = gnn.SchNet(mean=mean_ex,std=std_ex)
        self.sch_ex_2 = gnn.SchNet(mean=mean_ex,std=std_ex)
        self.sch_g_1 = gnn.SchNet(mean=mean_g,std=std_g)
        self.sch_g_2 = gnn.SchNet(mean=mean_g,std=std_g)

        self.mlp = nn.Linear(4,2)

        self.double()


    def forward(self,batch):
        x, edge_index = batch.x, batch.edge_index
        pos_g, pos_ex = batch.pos_g,batch.pos_ex
        x = x.long().squeeze()

        v1 = self.sch_ex_1(x,pos_ex,batch.batch)
        v2 = self.sch_ex_2(x,pos_ex,batch.batch)
        v3 = self.sch_g_1(x,pos_g,batch.batch)
        v4 = self.sch_g_2(x,pos_g,batch.batch)

        _x = torch.cat((v1,v2,v3,v4),axis=-1)
        _x = self.mlp(_x)
        return _x
