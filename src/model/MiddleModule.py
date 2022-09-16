import pytorch_lightning as pl
import torch

import pandas as pd
from datetime import datetime
import pathlib


class MiddleModule1(pl.LightningModule):

    """
    Module for code reuse.
    """

    def __init__(self,
            name : str = "noname",
            mean_g : float = None,
            std_g : float = None,
            mean_ex : float = None,
            std_ex : float = None,
            out_directory : str = "./"
        ):
        super().__init__()

        self.name = name
        self.mean_g = mean_g
        self.std_g = std_g
        self.mean_ex = mean_ex
        self.std_ex = std_ex

        if type(out_directory) == str:
            out_directory = pathlib.Path(out_directory)
        self.out_directory = out_directory

        self.mseloss = torch.nn.MSELoss()


    def forward(self,batch):
        raise NotImplementedError

    def training_step(self,batch,batch_idx):
        y_g, y_ex = batch.y_g, batch.y_ex

        if self.mean_g is not None and self.std_g is not None:
            y_g = (batch.y_g - self.mean_g) / self.std_g

        if self.mean_ex is not None and self.std_ex is not None:
            y_ex = (batch.y_ex - self.mean_ex) / self.std_ex

        y_hat = self(batch) # g, ex

        y_hat = torch.cat((y_hat[:,0],y_hat[:,1]))
        y = torch.cat((y_g,y_ex))
        loss = self.mseloss(y_hat.squeeze(),y)
        return loss

    def validation_step(self,batch,batch_idx):
        y_g, y_ex = batch.y_g, batch.y_ex
        
        y_hat = self(batch)

        if self.mean_g is not None and self.std_g is not None:
            y_hat[:,0] = y_hat[:,0]*self.std_g + self.mean_g

        if self.mean_ex is not None and self.std_ex is not None:
            y_hat[:,1] = y_hat[:,1]*self.std_ex + self.mean_ex

        y_hat = torch.cat((y_hat[:,0],y_hat[:,1]))
        y = torch.cat((batch.y_g,batch.y_ex))
        loss = self.mseloss(y_hat.squeeze(),y)
        loss = torch.sqrt(loss)*100

        return {'RMSEx100':loss.detach(),'n_rows':y_hat.shape[0]}

    def validation_epoch_end(self,outputs):
        sum_square_err = 0
        n_rows = 0
        for output in outputs:
            sum_square_err += ((output['RMSEx100']/100)**2)*output['n_rows']
            n_rows += output['n_rows']

        total_metric = torch.sqrt(sum_square_err / n_rows)*100

        print(f"Epoch {self.current_epoch} RMSEx100:{total_metric}")
        self.log("RMSEx100",total_metric ,on_step=False, on_epoch=True)

        return {"RMSEx100" : total_metric}

    def test_step(self,batch,batch_idx):
        y_hat = self(batch).detach().cpu()

        if self.mean_g is not None and self.std_g is not None:
            y_hat[:,0] = y_hat[:,0]*self.std_g + self.mean_g

        if self.mean_ex is not None and self.std_ex is not None:
            y_hat[:,1] = y_hat[:,1]*self.std_ex + self.mean_ex

        return {
            'Reorg_g' : y_hat[:,0],
            'Reorg_ex' : y_hat[:,1], 
            'index' : batch['index']
        }
    
    def test_epoch_end(self,outputs):

        result = {
            'Reorg_g' : torch.cat([x['Reorg_g'] for x in outputs],axis=0),
            'Reorg_ex': torch.cat([x['Reorg_ex'] for x in outputs],axis=0),
            'index': []
        }

        for x in outputs:
            result['index'].extend(x['index'])

        df_submission = pd.DataFrame.from_dict(result)
        df_submission.set_index('index',inplace=True)
        df_submission.to_csv(self.out_directory / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                  index_label="index")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.0001, last_epoch=-1)
        
        return [optimizer],[lr_scheduler]
   