import h5py
import numpy as np


def __peak_normalization(data):
    norm = np.array([absmax for absmax in np.abs(data).max()])
    data /= norm
    return data


def openhd5(path):
    data = h5py.File(path, 'r')
    return data


if __name__ == '__main__':
    path = '/vol/data/2025_wristus_wiicontroller_leitner/processed/Sequence_TWi0005_TSt0002_SWi0010/P0001/S0014_P0001_E0000_Xy.h5'
    data = openhd5(path)
    print()