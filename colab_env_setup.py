import numpy as np
import argparse
import os
import pathlib
import subprocess

def main(
        zip_file_path : pathlib.Path,
        dest_path : pathlib.Path
    ):

    dest_file_path = dest_path / zip_file_path.name

    commands = [
        ['cp',str(zip_file_path),str(dest_path)],
        ['unzip','-q',str(dest_file_path),'-d',str(dest_path)],
        ['pip','install','ase'],
        ['pip','install','torch-scatter','torch-sparse','torch-cluster',
            'torch-spline-conv','torch-geometric','-f','https://data.pyg.org/whl/torch-1.12.0+cu113.html'],
        ['pip','install','pytorch-lightning']
    ]

    for cmd in commands:
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("Something went wrong!")
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Copy data files in google drive to colab local storage")
    parser.add_argument("--zip_path",dest = 'zip_file',type=pathlib.Path,default=pathlib.Path("./data/data.zip"))
    parser.add_argument("--dest_path",dest= 'destination',type=pathlib.Path,default=pathlib.Path("/content"))


    args = parser.parse_args()

    main(args.zip_file,args.destination)