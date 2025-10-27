import os

import pandas as pd
from omegaconf import OmegaConf
from git import Repo

def get_git_root(path=os.getcwd()):
    repo = Repo(path, search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")

def load_config(config_file):
    cfg = OmegaConf.load(config_file)
    return cfg

def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

