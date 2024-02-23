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

  
filelist = '../../TopNN/data/H5/list_all.txt'

def get_idxmap(filelist,dataset='train'):
    idxmap = {}
    offset = 0 
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            print('idxmap: ',filename)
            with h5py.File(filename, 'r') as Data:
                length = len(Data['labels'][:])
                if dataset=='train': length = int(length*0.9)
                if dataset=='val': length = int(length*0.05)
                idxmap[filename] = np.arange(offset,offset+int(length))
                offset += int(length)
    return idxmap

def create_integer_file_map(idxmap):
    integer_file_map = {}
    file_names = list(idxmap.keys())
    file_vectors = list(idxmap.values())
    for i, file in enumerate(file_names):
        print('integer_file_map: ',file)
        vector = file_vectors[i]
        for integer in vector:
            if integer in integer_file_map:
                integer_file_map[integer].append(file)
            else:
                integer_file_map[integer] = [file]

    return integer_file_map


idxmap = get_idxmap(filelist,dataset='train')
idxmap_val = get_idxmap(filelist,dataset='val')
integer_file_map = create_integer_file_map(idxmap)
integer_file_map_val = create_integer_file_map(idxmap_val)