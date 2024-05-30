from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import uproot
import awkward as ak
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from sklearn.preprocessing import StandardScaler
import pickle
import h5py

  

if __name__ == "__main__":
    filelist = '/raven/u/mvigl/Stop/TopNN/data/H5/test/list_all.txt'
    with h5py.File('/raven/u/mvigl/Stop/data/baseline_H5_test/Virtual_test.h5',mode='w') as h5fw:
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                data_index = (filename.index("/mc"))
                name = (filename[data_index+1:])
                if 'MCRun2_Signal_AF3' in filename: name = 'MCRun2_Signal_AF3_' + name
                if 'MCRun2_Signal_FS' in filename: name = 'MCRun2_Signal_FS_' + name 
                print(name)
                h5fw[name] = h5py.ExternalLink(filename,'/')   


