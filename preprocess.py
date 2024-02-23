import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='/raven/u/mvigl/Stop/TopNN/data/list_all.txt')
args = parser.parse_args()

def split_data(length,array,dataset='train'):
    idx_train = int(length*0.9)
    idx_val = int(length*0.95)
    if dataset=='train': return array[:idx_train]
    if dataset=='val': return array[idx_train:idx_val]    
    if dataset=='test': return array[idx_val:]    
    if dataset=='full': return array[:]  
    else:       
        print('choose: train, val, test')
        return 0       

def idxs_to_var(out,branches,var,dataset):
    length = len(ak.Array(branches['ljetIdxs_saved'][:]))

    out['label'] = (ak.flatten(  split_data(length,ak.Array(branches['multiplets']),dataset=dataset)  )[:,-1].to_numpy()+1)/2
    bj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,0]))[:,:,0]*(ak.Array(branches['bjet1'+var]))
    bj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,1]))[:,:,0]*(ak.Array(branches['bjet2'+var]))
    out['bjet_'+var] = ak.flatten(   split_data(length,(bj1 + bj2),dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,1]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,1]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,1]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,1]*(ak.Array(branches['ljet4'+var]))
    out['jet1_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,2]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,2]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,2]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,2]*(ak.Array(branches['ljet4'+var]))
    out['jet2_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),dataset=dataset)   ).to_numpy()

    return out

def get_data(branches,vars=['pT','eta','phi','M'],dataset='train'):
    output = {}
    for var in vars:
        output = idxs_to_var(output,branches,var,dataset)
        
    out_data = np.hstack(   (   output['bjet_pT'][:,np.newaxis],
                                output['jet1_pT'][:,np.newaxis],
                                output['jet2_pT'][:,np.newaxis],
                                output['bjet_eta'][:,np.newaxis],
                                output['jet1_eta'][:,np.newaxis],
                                output['jet2_eta'][:,np.newaxis],
                                output['bjet_phi'][:,np.newaxis],
                                output['jet1_phi'][:,np.newaxis],
                                output['jet2_phi'][:,np.newaxis],
                                output['bjet_M'][:,np.newaxis],
                                output['jet1_M'][:,np.newaxis],
                                output['jet2_M'][:,np.newaxis]
                            ) 
                        )
                   
    return out_data,output['label']


Features = ['multiplets',
            'bjetIdxs_saved',
            'ljetIdxs_saved',
            'bjet1pT',
            'bjet2pT',
            'ljet1pT',
            'ljet2pT',
            'ljet3pT',
            'ljet4pT',
            'bjet1eta',
            'bjet2eta',
            'ljet1eta',
            'ljet2eta',
            'ljet3eta',
            'ljet4eta',
            'bjet1phi',
            'bjet2phi',
            'ljet1phi',
            'ljet2phi',
            'ljet3phi',
            'ljet4phi',
            'bjet1M',
            'bjet2M',
            'ljet1M',
            'ljet2M',
            'ljet3M',
            'ljet4M',
            ]

with open(args.filelist) as f:
    for line in f:
        filename = line.strip()
        print('reading : ',filename)
        with uproot.open({filename: "stop1L_NONE;1"}) as tree:
            branches = tree.arrays(Features)
            multiplets,labels = get_data(branches,dataset='full')
            out_dir = '/raven/u/mvigl/Stop/data/H5_full'
            if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
            print('reading : ',filename)
            data_index = filename.index("/mc")
            out_dir = out_dir + (filename[data_index:]).replace(".root",".h5")
            with h5py.File(out_dir, 'w') as out_file: 
                out_file.create_dataset('multiplets', data=multiplets)
                out_file.create_dataset('labels', data=labels.reshape(-1),dtype='i4')


    
