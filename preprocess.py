import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os
import yaml

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='list_all_mc20.txt')
parser.add_argument('--split', help='train,test,val',default='odd')
parser.add_argument('--out_dir', help='out_dir',default='/raven/u/mvigl/Stop/run/pre/H5_samples')
args = parser.parse_args()

def split_data(length,array,rnum_array,sig=True,dataset='test',sigsub=1):
    if dataset == 'odd': return array[rnum_array%2==1]
    elif dataset == 'even': return array[rnum_array%2==0]
    else: 
        idx_train = int(length*0.95)
        if dataset=='train': 
            if (sigsub!=1 and sig): return array[:int(length*0.95*sigsub)] 
            else: return array[:idx_train] 
        if dataset=='test': return array[idx_train:]    
        if dataset=='full': return array[:]  
        else:       
            print('choose: train, val, test')
            return 0        

def idxs_to_var(out,branches,var,dataset):
    length = len(ak.Array(branches['ljetIdxs_saved'][:]))
    rnum_array = (ak.Array(branches['RunNumber'])[:])
    counts = ak.num( split_data(length,ak.Array(branches['multiplets']),rnum_array=rnum_array,dataset=dataset) )

    out['label'] = (ak.flatten(  split_data(length,ak.Array(branches['multiplets']),rnum_array=rnum_array,dataset=dataset)  )[:,-1].to_numpy()+1)/2
    bj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,0]))[:,:,0]*(ak.Array(branches['bjet1'+var]))
    bj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,1]))[:,:,0]*(ak.Array(branches['bjet2'+var]))
    out['bjet_'+var] = ak.flatten(   split_data(length,(bj1 + bj2),rnum_array=rnum_array,dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,1]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,1]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,1]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,1]*(ak.Array(branches['ljet4'+var]))
    out['jet1_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),rnum_array=rnum_array,dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,2]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,2]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,2]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,2]*(ak.Array(branches['ljet4'+var]))
    out['jet2_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),rnum_array=rnum_array,dataset=dataset)   ).to_numpy()

    truth_info = {
        'counts': counts,
        'truth_top_min_dR': split_data(length,branches['truth_top_min_dR'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_top_min_dR_m': split_data(length,branches['truth_top_min_dR_m'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_top_min_dR_jj': split_data(length,branches['truth_top_min_dR_jj'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_top_min_dR_m_jj': split_data(length,branches['truth_top_min_dR_m_jj'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_topp_match': split_data(length,branches['truth_topp_match'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_topm_match': split_data(length,branches['truth_topm_match'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_topp_pt': split_data(length,branches['truth_topp_pt'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_topm_pt': split_data(length,branches['truth_topm_pt'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_Wp_pt': split_data(length,branches['truth_Wp_pt'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'truth_Wm_pt': split_data(length,branches['truth_Wm_pt'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
        'WeightEvents': split_data(length,branches['WeightEvents'][:].to_numpy(),rnum_array=rnum_array,dataset=dataset),
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
            'truth_top_min_dR',
            'truth_top_min_dR_m',
            'truth_top_min_dR_jj',
            'truth_top_min_dR_m_jj',
            'truth_topp_match',
            'truth_topm_match',
            'truth_topp_pt',
            'truth_topm_pt',
            'truth_Wp_pt',
            'truth_Wm_pt',
            'WeightEvents',
            'RunNumber'
            ]

inputs = [  'bjet_pT',
            'jet1_pT',
            'jet2_pT',
            'bjet_eta',
            'jet1_eta',
            'jet2_eta',
            'bjet_phi',
            'jet1_phi',
            'jet2_phi',
            'bjet_M',
            'jet1_M',
            'jet2_M',
]

variables = ['counts',
            'truth_top_min_dR',
            'truth_top_min_dR_m',
            'truth_top_min_dR_jj',
            'truth_top_min_dR_m_jj',
            'truth_topp_match',
            'truth_topm_match',
            'truth_topp_pt',
            'truth_topm_pt',
            'truth_Wp_pt',
            'truth_Wm_pt',
            'WeightEvents',
            ]

dataset = args.split
with open(args.filelist) as f:
    i=0
    out_dir = f'{args.out_dir}_{dataset}/'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    for line in f:
        filename = line.strip()
        print('reading : ',filename)
        number = number = filename[(filename.index("TeV.")+4):(filename.index(".stop1L"))]
        with open('/raven/u/mvigl/Stop/TopNN/data/stop_masses.yaml') as g:
            map = yaml.load(g, Loader=yaml.FullLoader)['samples'] 
        if number not in map.keys(): 
            print('--- skipping sample ---')
            continue
        with uproot.open({filename: "stop1L_NONE;1"}) as tree:
            if i==0:
                branches = tree.arrays(Features)
                multiplets,labels,out_truth_info = get_data(branches,dataset=dataset)
            else:
                branches = tree.arrays(Features)
                multiplets_i,labels_i,out_truth_info_i = get_data(branches,dataset=dataset)
                multiplets=np.concatenate((multiplets,multiplets_i),axis=0)
                labels=np.concatenate((labels,labels_i),axis=0)
                out_truth_info=np.concatenate((out_truth_info,out_truth_info_i),axis=0)
        i+=1
    with h5py.File(f'{out_dir}multiplets_{dataset}.h5', 'w') as out_file: 
        out_file.create_dataset('multiplets', data=multiplets,compression="gzip")
        out_file.create_dataset('labels', data=labels.reshape(-1),dtype='i4',compression="gzip")
        out_file.create_dataset('variables', data=out_truth_info,compression="gzip")