import numpy as np
import pathlib
import argparse
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from typing import Optional, Union

from molreader import *

def npmol_to_data(node_dat, edge_dat):
    nodes = node_dat['proton_no'].copy().astype(int)

    node_position = np.stack((node_dat['x'],node_dat['y'],node_dat['z']),axis=-1)
    
    v_tail = np.concatenate([edge_dat['v1'],edge_dat['v2']])
    v_head = np.concatenate([edge_dat['v2'],edge_dat['v1']])

    edge_index = np.stack((v_tail,v_head))

    edge_attr = np.zeros((edge_dat.shape[0],2),dtype=float)
    edge_attr[:,0] = edge_dat['conn_type']

    # get distance between 2 atom
    pos_ordered_v1 = node_position[edge_dat['v1']]
    pos_ordered_v2 = node_position[edge_dat['v2']]
    edge_len = np.linalg.norm((pos_ordered_v1-pos_ordered_v2),axis=1)
    edge_attr[:,1]=edge_len

    edge_attr = np.concatenate((edge_attr,edge_attr),axis=0)

    geo_dat = Data(
        x=torch.tensor(nodes,dtype=torch.int),
        edge_index=torch.tensor(edge_index,dtype=torch.long),
        edge_attr=torch.tensor(edge_attr,dtype=torch.float),
        pos = torch.tensor(node_position,dtype=torch.float),
    )
    return geo_dat

def proc_file(
        file_path : Union[pathlib.Path, str],
        dest_path : Optional[Union[pathlib.Path, str, None]] = None
    ):
    if type(file_path) == str :
        file_path = pathlib.Path(file_path)
    if type(dest_path) == str:
        dest_path = pathlib.Path(dest_path)

    node_dat, edge_dat = mol_to_dict(file_path)

    # fix to index starting from 0 
    edge_dat['v1'] = edge_dat['v1'] -1 
    edge_dat['v2'] = edge_dat['v2'] -1 
    
    geo_dat = npmol_to_data(node_dat,edge_dat)

    if dest_path:
        torch.save(geo_dat,dest_path)

    return geo_dat

def proc_folder(
        folder_path : Union[pathlib.Path, str],
        dest_folder_path : Optional[Union[pathlib.Path, str, None]] = None,
        ret : bool = False,
        use_tqdm : bool = False
    ):
    if type(folder_path) == str :
        folder_path = pathlib.Path(folder_path)
    if type(dest_folder_path) == str :
        dest_folder_path = pathlib.Path(dest_folder_path)

    file_paths = sorted(list(folder_path.glob("*.mol")))
    geo_dats = []

    if dest_folder_path :
        dest_folder_path.mkdir(parents=True,exist_ok=True)

    path_enumerator = tqdm(file_paths) if use_tqdm else file_paths

    for file_path in path_enumerator:
        file_dest_path = dest_folder_path / (file_path.stem+".pt") if dest_folder_path else None
        geo_dat = proc_file(file_path,file_dest_path)

        if ret:
            geo_dats.append(geo_dat)

    if ret:
        return geo_dats

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('source_folder',type=pathlib.Path)
    parser.add_argument('dest_folder',type=pathlib.Path)

    args = parser.parse_args()

    proc_folder(
        args.source_folder,
        args.dest_folder,
        use_tqdm=True
    )

