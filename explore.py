import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os



filelist = '/raven/u/mvigl/Stop/TopNN/data/H5/list_all.txt'
with open(filelist) as f:
    for line in f:
        filename = line.strip()
        print('reading : ',filename)
        with h5py.File(filename, 'r') as f:
            labels = f['labels'][:]
            odd_labels = np.sum(labels==0.5)
            pos_labels = np.sum(labels==1)
            neg_labels = np.sum(labels==0)
            all_labels = len(labels)
            print('odd labels: ', odd_labels, ' out of: ', all_labels, ' meaning: ',(odd_labels/all_labels)*100, 'percent' )
            print('pos labels: ', pos_labels, ' out of: ', all_labels, ' meaning: ',(pos_labels/all_labels)*100, 'percent' )
            print('neg labels: ', neg_labels, ' out of: ', all_labels, ' meaning: ',(neg_labels/all_labels)*100, 'percent' )


    
