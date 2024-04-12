import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='train_list_testing.txt')
parser.add_argument('--split', help='train,test,val',default='train')
parser.add_argument('--out_dir', help='out_dir',default='H5_spanet')
args = parser.parse_args()

def split_data(length,array,dataset='test'):
    idx_train = int(length*0.95)
    if dataset=='train': return array[:idx_train] 
    if dataset=='test': 
        if length-idx_train> 200000: return array[length-200000:] 
        else: return array[idx_train:]    
    if dataset=='full': return array[:]  
    else:       
        print('choose: train, val, test')
        return 0       

def idxs_to_var(branches,dataset):
    filter =  (ak.Array(branches['multiplets'])[:,0,-1]==1)
    length = len(filter)
    vars=['btag','eta','M','phi','pT']
    inputs = {}
    for var in vars:
        if var == 'btag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,0] += 1
            inputs[var][:,1] += 1
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
        else:
            inputs[var] = np.zeros((length,10))
            print(ak.Array(branches['bjet1'+var][filter]).to_numpy())
            inputs[var][:,0] += ak.Array(branches['bjet1'+var][filter]).to_numpy()
            inputs[var][:,1] += ak.Array(branches['bjet2'+var][filter]).to_numpy()
            inputs[var][:,2] += ak.Array(branches['ljet1'+var][filter]).to_numpy()
            inputs[var][:,3] += ak.Array(branches['ljet2'+var][filter]).to_numpy()
            inputs[var][:,4] += ak.Array(branches['ljet3'+var][filter]).to_numpy()
            inputs[var][:,5] += ak.Array(branches['ljet4'+var][filter]).to_numpy()
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
    mask = (inputs['pT']<=0).astype(int)    

    targets = {}
    targets['htb'] = -np.ones((length))
    targets['q1'] = -np.ones((length))
    targets['q2'] = -np.ones((length))
    targets['ltb'] = -np.ones((length))

    targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,0]))] = 0
    targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,0]))] = 1
    targets['ltb'] = np.abs(targets['htb']-1)
    targets['q1'][(ak.Array(branches['multiplets'][filter,1,0])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
    targets['q1'][(ak.Array(branches['multiplets'][filter,1,0])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
    targets['q1'][(ak.Array(branches['multiplets'][filter,1,0])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
    targets['q1'][(ak.Array(branches['multiplets'][filter,1,0])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5
    targets['q2'][(ak.Array(branches['multiplets'][filter,2,0])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
    targets['q2'][(ak.Array(branches['multiplets'][filter,2,0])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
    targets['q2'][(ak.Array(branches['multiplets'][filter,2,0])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
    targets['q2'][(ak.Array(branches['multiplets'][filter,2,0])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5

    targets['htb'] = split_data(length,targets['htb'],dataset=dataset)
    targets['ltb'] = split_data(length,targets['ltb'],dataset=dataset)
    targets['q1'] = split_data(length,targets['q1'],dataset=dataset)
    targets['q2'] = split_data(length,targets['q2'],dataset=dataset)

    truth_info = {
        'truth_top_min_dR': split_data(length,branches['truth_top_min_dR'][filter].to_numpy(),dataset=dataset),
        'truth_top_min_dR_m': split_data(length,branches['truth_top_min_dR_m'][filter].to_numpy(),dataset=dataset),
        'truth_top_min_dR_jj': split_data(length,branches['truth_top_min_dR_jj'][filter].to_numpy(),dataset=dataset),
        'truth_top_min_dR_m_jj': split_data(length,branches['truth_top_min_dR_m_jj'][filter].to_numpy(),dataset=dataset),
        'truth_topp_match': split_data(length,branches['truth_topp_match'][filter].to_numpy(),dataset=dataset),
        'truth_topm_match': split_data(length,branches['truth_topm_match'][filter].to_numpy(),dataset=dataset),
        'truth_topp_pt': split_data(length,branches['truth_topp_pt'][filter].to_numpy(),dataset=dataset),
        'truth_topm_pt': split_data(length,branches['truth_topm_pt'][filter].to_numpy(),dataset=dataset),
        'truth_Wp_pt': split_data(length,branches['truth_Wp_pt'][filter].to_numpy(),dataset=dataset),
        'truth_Wm_pt': split_data(length,branches['truth_Wm_pt'][filter].to_numpy(),dataset=dataset),
        'WeightEvents': split_data(length,branches['WeightEvents'][filter].to_numpy(),dataset=dataset),
    }
    return mask,inputs,targets,truth_info

def get_data(branches,vars=['eta','M','phi','pT'],dataset='train'):
    mask,inputs,targets,truth_info = idxs_to_var(branches,dataset)
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
                   
    return mask,inputs,targets,out_truth_info


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
            'WeightEvents'
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
out_dir = f'{args.out_dir}_{dataset}/'
with open(args.filelist) as f:
    for line in f:
        filename = line.strip()
        print('reading : ',filename)
        with uproot.open({filename: "stop1L_NONE;1"}) as tree:
            branches = tree.arrays(Features)
            mask,inputs,targets,out_truth_info = get_data(branches,dataset=dataset)
            if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
            data_index = filename.index("/MC")
            out_dir = out_dir + (filename[data_index:]).replace(".root",f"_{dataset}.h5")
            with h5py.File(out_dir, 'w') as out_file: 
                out_file.create_dataset('INPUTS', data=inputs,compression="gzip")
                inputs_group = out_file.create_group('INPUTS')
                source = inputs_group.create_group(f'SOURCE')
                source.create_dataset('MASK', data=mask)
                source.create_dataset('btag', data=inputs['btag'])
                source.create_dataset('eta', data=inputs['eta'])
                source.create_dataset('mass', data=inputs['mass'])
                source.create_dataset('phi', data=inputs['phi'])
                source.create_dataset('pt', data=inputs['pt'])
                targets_group = out_file.create_group('TARGETS')
                ht = targets_group.create_group(f'ht')
                ht.create_dataset('b', data=targets['htb'])
                ht.create_dataset('q1', data=targets['q1'])
                ht.create_dataset('q2', data=targets['q2'])
                lt = targets_group.create_group(f'lt')
                ht.create_dataset('b', data=targets['ltb'])