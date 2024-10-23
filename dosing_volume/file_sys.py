"""
The file control library.
Including 1. load/dump pickle, json ... 2. useful work directories
TODO: 1. Support yaml dump
"""
import os
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import marshal

# some path for working directories
workdir = Path(__file__).parent
workdir_model = workdir / "models"
workdir_vision = workdir / "vision"
workdir_core = workdir / "core"
workdir_learning = workdir / "learning"
workdir_data = workdir / "data"


def dump_pickle(data, path="", reminder=True):
    """Dump the data by pickle"""
    path = str(path)
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return True


def load_pickle(path):
    """Load the data by pickle"""
    path = str(path)
    if os.path.exists(path):
        pass
    else:
        raise Exception(f"The file {path} is not exist")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def dump_marshal(data, path="", reminder=True):
    """Dump the data by marshal"""
    path = str(path)
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "wb") as f:
        marshal.dump(data, f)
    return True


def load_marshal(path):
    """Load the data by marshal"""
    path = str(path)
    if os.path.exists(path):
        pass
    else:
        raise Exception(f"The file {path} is not exist")
    with open(path, "rb") as f:
        data = marshal.load(f)
    return data


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


def dump_ply(data, path="", reminder=True):
    """Dump the data to ply"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(path, pcd)
    return True


def load_ply(path):
    path = str(path)
    """Load the data by ply"""
    if os.path.exists(path):
        pass
    else:
        raise Exception(f"The file {path} is not exist")
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def load_json(path):
    """Load the data by json"""
    if os.path.exists(path):
        pass
    else:
        raise Exception(f"The file {path} is not exist")
    with open(path, "rb") as f:
        data = json.load(f)
    return data


def get_filename(path):
    """Get the filename"""
    return Path(path).name


def get_filename_we(path):
    """Get the filename without extension"""
    return ".".join(Path(path).name.split(".")[:-1])