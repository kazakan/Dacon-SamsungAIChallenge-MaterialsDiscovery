import pathlib

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from typing import Union, Optional

import pytorch_lightning as pl
import pandas as pd

class ReorgEnergyDataset(Dataset):

    def __init__(self,
        csv_path : Union[pathlib.Path, str],
        dir_path : Union[pathlib.Path, str],
        train=False
        ):
        super().__init__()

        if type(dir_path) is str:
            dir_path = pathlib.Path(dir_path)

        if type(csv_path) is str:
            csv_path = pathlib.Path(csv_path)

        self.dir_path = dir_path
        self.csv_path = csv_path


        df_dat = pd.read_csv(csv_path)

        self.index = df_dat['index'].values
        if train:
            self.y = torch.tensor(df_dat[['Reorg_g','Reorg_ex']].values) 
        else : 
            # just for compatibility
            self.y = torch.zeros((len(self.index),2))

    def __len__(self):
        return len(self.index)

    def __getitem__(self,idx):
        dat_g = torch.load(self.dir_path / (self.index[idx]+"_g.pt"))
        dat_ex = torch.load(self.dir_path / (self.index[idx]+"_ex.pt"))

        dat = Data(
            x=dat_g.x,
            edge_index = dat_g.edge_index,
            y_g = self.y[idx,0],
            y_ex = self.y[idx,1],
            pos_g = dat_g.pos,
            pos_ex = dat_ex.pos,
            edge_attr_g = dat_g.edge_attr,
            edge_attr_ex = dat_ex.edge_attr,
            index = self.index[idx]
        ) 

        return dat
                


class ReorgEnergyDataModule(pl.LightningDataModule):

    def __init__(self, 
            train_dir : str,
            train_csv_path : str,
            test_dir : str, 
            test_csv_path : str,
            batch_size : int =32
        ):
        super().__init__()

        self.train_dir = train_dir
        self.train_csv_path = train_csv_path
        self.test_dir = test_dir
        self.test_csv_path = test_csv_path

        self.batch_size = batch_size

    def setup(self,stage : str):
        data = ReorgEnergyDataset(self.train_csv_path,self.train_dir,train=True)
        l1 = int(len(data)*0.8)
        l2 = len(data) - l1

        self.train_set, self.val_set = random_split(data,[l1,l2],
            generator=torch.Generator().manual_seed(42)
        )

        self.test_set = ReorgEnergyDataset(self.test_csv_path, self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size = self.batch_size)

    