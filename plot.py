import numpy as np
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import torch

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
#device = get_device()
device = 'cpu'

def split_data(length,array,dataset='train'):
    idx_train = int(length*0.9)
    idx_val = int(length*0.95)
    if dataset=='train': return array[:idx_train]
    if dataset=='val': return array[idx_train:idx_val]    
    if dataset=='test': return array[idx_val:]    
    else:       
        print('choose: train, val, test')
        return 0       

def idxs_to_var(out,branches,var,dataset):
    length = len(ak.Array(branches['ljetIdxs_saved'][:]))
    counts = ak.num( split_data(length,ak.Array(branches['multiplets']),dataset=dataset) )

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

    truth_info = {
        'counts': counts,
        'truth_top_min_dR': split_data(length,branches['truth_top_min_dR'][:].to_numpy(),dataset=dataset),
        'truth_top_min_dR_m': split_data(length,branches['truth_top_min_dR_m'][:].to_numpy(),dataset=dataset),
        'truth_top_min_dR_jj': split_data(length,branches['truth_top_min_dR_jj'][:].to_numpy(),dataset=dataset),
        'truth_top_min_dR_m_jj': split_data(length,branches['truth_top_min_dR_m_jj'][:].to_numpy(),dataset=dataset),
        'truth_topp_match': split_data(length,branches['truth_topp_match'][:].to_numpy(),dataset=dataset),
        'truth_topm_match': split_data(length,branches['truth_topm_match'][:].to_numpy(),dataset=dataset),
        'truth_topp_pt': split_data(length,branches['truth_topp_pt'][:].to_numpy(),dataset=dataset),
        'truth_topm_pt': split_data(length,branches['truth_topm_pt'][:].to_numpy(),dataset=dataset),
        'truth_Wp_pt': split_data(length,branches['truth_Wp_pt'][:].to_numpy(),dataset=dataset),
        'truth_Wm_pt': split_data(length,branches['truth_Wm_pt'][:].to_numpy(),dataset=dataset),
        'WeightEvents': split_data(length,branches['WeightEvents'][:].to_numpy(),dataset=dataset),
    }
    return out,truth_info

def get_data(branches,vars=['pT','eta','phi','M'],dataset='train'):
    output = {}
    for var in vars:
        output,truth_info = idxs_to_var(output,branches,var,dataset)
        
    out_data = np.hstack(   (       output['bjet_pT'][:,np.newaxis],
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
    out_truth_info = np.hstack(   (     truth_info['counts'][:,np.newaxis],
                                        truth_info['truth_top_min_dR'][:,np.newaxis],
                                        truth_info['truth_top_min_dR_m'][:,np.newaxis],
                                        truth_info['truth_top_min_dR_jj'][:,np.newaxis],
                                        truth_info['truth_top_min_dR_m_jj'][:,np.newaxis],
                                        truth_info['truth_topp_match'][:,np.newaxis],
                                        truth_info['truth_topm_match'][:,np.newaxis],
                                        truth_info['truth_topp_pt'][:,np.newaxis],
                                        truth_info['truth_topm_pt'][:,np.newaxis],
                                        truth_info['truth_Wp_pt'][:,np.newaxis],
                                        truth_info['truth_Wm_pt'][:,np.newaxis],
                                        truth_info['WeightEvents'][:,np.newaxis]
                            ) 
                        )
                   
    return out_data,output['label'],out_truth_info


    
