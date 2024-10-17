import zipfile
import pickle as pkl
import wandb
import pandas as pd
from typing import Any
import subprocess

drop_features_conf_wandb = {
    'keep_ratio' : 0.3,
    'rand_drop_ratio' : 0.2,
    'rand_drop_rows_threshold' : 10,
    'random_seed' : 42
    }

def read_obj(name):
    file = open(name, "rb")
    obj = pkl.load(file)
    file.close()

    return obj


def evals_logger(set_evals_df: pd.DataFrame, set_eval_dict:dict[str,Any], name:str):
    my_table = wandb.Table(dataframe = set_evals_df)
    wandb.log({"folds_evals_" + name: my_table})
    wandb.log(set_eval_dict)


def fetch_artifacts(run_filter):
    metadata_dict = {}
    api = wandb.Api()
    runs = api.runs(**run_filter)
    for run in runs:
        for artifact in run.logged_artifacts():
            for file in artifact.files():
                try:
                    if ".pkl" in file.name:
                        print("file name: ", file.name)
                        artifact.download()
                        metadata_dict[f"{file.name}"] = artifact.metadata
                except Exception as e:
                    print("ERORR")
                    print(file.name)
                    print(artifact.metadata)
                    print(e)
                    pass
    return metadata_dict


def unzip(path):
    import zipfile

    with zipfile.ZipFile(path, "r") as zipf:
        name = ".".join(path.split("/")[-1].split(".")[:-1])
        folder = "/".join(path.split("/")[:-1])

        zipf.extract(name, folder)


def read_tracker_objects(path="/kaggle/working/artifacts", metadata_dict=None):
    import os

    objects = {}
    for path, _, files in os.walk(path):
        for name in files:
            pth = os.path.join(path, name)
            if ".zip" in pth and 'dataset' not in pth:
                unzip(pth)
                pth = ".".join(pth.split(".")[:-1])
                obj_name = ".".join(name.split(".")[:-2])
                if metadata_dict is not None:
                    objects[obj_name] = {
                        "obj": read_obj(pth),
                        "metadata": metadata_dict[name],
                    }
                else:
                    objects[obj_name] = {"obj": read_obj(pth), "metadata": {}}
    return objects



def download_wandb_artifact(wandb_api,artifact_path,save_path, print_metadata = False):
    """
    download artifact from wandb with artifact_path as input and save_path for save the file.
    
    """
    dataset_path = save_path
    art = wandb_api.artifact(artifact_path)
    if print_metadata:
        print('Metadata: ',art.metadata)
    dir = art.download(dataset_path)
    
    pth = save_path.split('/')[1]
    # Run the ls command using subprocess and capture the output
    result = subprocess.run(["ls", pth], stdout=subprocess.PIPE, text=True)
    # Get the list of files from the output
    file_list = result.stdout.splitlines()
    print("Artifact files:", file_list)