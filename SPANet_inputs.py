import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='data/root/list_sig_FS_testing.txt')
parser.add_argument('--split', help='train,test,val',default='train')
parser.add_argument('--out_dir', help='out_dir',default='SPANet_all_8_cat')
args = parser.parse_args()

def split_data(length,array,dataset='test'):
    idx_train = int(length*0.95)
    if dataset=='train': return array[:idx_train] 
    if dataset=='test': return array[idx_train:]    
        #if length-idx_train> 200000: return array[length-200000:] 
        #else: return array[idx_train:]    
    if dataset=='full': return array[:]  
    else:       
        print('choose: train, val, test')
        return 0       

def idxs_to_var(branches,dataset):
    #filter =  (ak.Array(branches['multiplets'])[:,0,-1]==1)
    #if dataset == 'test': 
    filter = (ak.Array(branches['multiplets'])[:,0,-1] > -100)
    not_matched =  (ak.Array(branches['multiplets'])[filter,0,-1]!=1)
    length = np.sum(filter)
    vars=['btag','qtag','etag','eta','M','phi','pT']
    inputs = {}
    for var in vars:
        if var == 'btag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,0] += 1
            inputs[var][:,1] += 1
            inputs[var][:,6] += 1
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
        elif var == 'qtag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,2] += 1
            inputs[var][:,3] += 1
            inputs[var][:,4] += 1
            inputs[var][:,5] += 1  
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
        elif var == 'etag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,7] += 1
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
        else:
            inputs[var] = np.zeros((length,10))
            inputs[var][:,0] += ak.Array(branches['bjet1'+var][filter]).to_numpy()
            inputs[var][:,1] += ak.Array(branches['bjet2'+var][filter]).to_numpy()
            inputs[var][:,2] += ak.Array(branches['ljet1'+var][filter]).to_numpy()
            inputs[var][:,3] += ak.Array(branches['ljet2'+var][filter]).to_numpy()
            inputs[var][:,4] += ak.Array(branches['ljet3'+var][filter]).to_numpy()
            inputs[var][:,5] += ak.Array(branches['ljet4'+var][filter]).to_numpy()
            inputs[var][:,7] += ak.Array(branches['lep1'+var][filter]).to_numpy()
            inputs[var][:,6] += ak.Array(branches['bjet3'+var][filter]).to_numpy()
            (inputs[var])[inputs[var]==-10]=0.
            inputs[var] = split_data(length,inputs[var],dataset=dataset)
    mask = (inputs['pT']>0)
    inputs['btag'][mask==False]=0.
    inputs['qtag'][mask==False]=0.
    inputs['etag'][mask==False]=0.

    targets = {}
    targets['htb'] = -np.ones((length))
    targets['q1'] = -np.ones((length))
    targets['q2'] = -np.ones((length))
    targets['ltb'] = -np.ones((length))
    targets['ltl'] = np.ones((length))*7

    targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,0]))] = 0
    targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,1]))] = 1
    targets['ltb'] = np.abs(targets['htb']-1)
    targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
    targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
    targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
    targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5
    targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
    targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
    targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
    targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5

    targets['htb'][not_matched] = -1
    targets['q1'][not_matched] = -1
    targets['q2'][not_matched] = -1
    targets['ltb'][not_matched] = -1
    targets['ltl'][not_matched] = -1

    targets['htb'] = split_data(length,targets['htb'],dataset=dataset)
    targets['q1'] = split_data(length,targets['q1'],dataset=dataset)
    targets['q2'] = split_data(length,targets['q2'],dataset=dataset)
    targets['ltb'] = split_data(length,targets['ltb'],dataset=dataset)
    single_b = ((mask[:,0]*mask[:,1])==False)
    (targets['ltb'])[single_b]=-1
    targets['ltl'] = split_data(length,targets['ltl'],dataset=dataset)
    targets['ltl'][(mask[:,7]==False)]=-1

    met = {
        'MET': split_data(length,branches['MET'][filter].to_numpy(),dataset=dataset),
        'METsig': split_data(length,branches['METsig'][filter].to_numpy(),dataset=dataset),
        'METphi': split_data(length,branches['METphi'][filter].to_numpy(),dataset=dataset),
        'MET_Soft': split_data(length,branches['MET_Soft'][filter].to_numpy(),dataset=dataset),
        'MET_Jet': split_data(length,branches['MET_Jet'][filter].to_numpy(),dataset=dataset),
        'MET_Ele': split_data(length,branches['MET_Ele'][filter].to_numpy(),dataset=dataset),
        'MET_Muon': split_data(length,branches['MET_Muon'][filter].to_numpy(),dataset=dataset),
        'mT_METl': split_data(length,branches['mT_METl'][filter].to_numpy(),dataset=dataset),
        'dR_bb': split_data(length,branches['dR_bb'][filter].to_numpy(),dataset=dataset),
        'dphi_METl': split_data(length,branches['dphi_METl'][filter].to_numpy(),dataset=dataset),
        'MT2_bb': split_data(length,branches['MT2_bb'][filter].to_numpy(),dataset=dataset),
        'MT2_b1l1_b2': split_data(length,branches['MT2_b1l1_b2'][filter].to_numpy(),dataset=dataset),
        'MT2_b2l1_b1': split_data(length,branches['MT2_b2l1_b1'][filter].to_numpy(),dataset=dataset),
        'MT2_min': split_data(length,branches['MT2_min'][filter].to_numpy(),dataset=dataset),
    }

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
        'is_matched': split_data(length,(branches['multiplets'][filter,0,-1]==1).to_numpy(),dataset=dataset)
    }
    return mask,inputs,targets,truth_info,met

def get_data(branches,vars=['eta','M','phi','pT'],dataset='train',sig=True):
    mask,inputs,targets,truth_info,met = idxs_to_var(branches,dataset)
    if sig:
        signal = np.ones(len(mask))
    else:
        signal = np.zeros(len(mask))    
    return mask,inputs,targets,truth_info,met,signal

def merge(d1,d2):
    merged_dict = {}
    for key in d1.keys():
        merged_dict[key] = np.concatenate((d1[key],d2[key]),axis=0)
    return merged_dict

Features = ['multiplets',
            'bjetIdxs_saved',
            'ljetIdxs_saved',
            'bjet1pT',
            'bjet2pT',
            'bjet3pT',
            'ljet1pT',
            'ljet2pT',
            'ljet3pT',
            'ljet4pT',
            'bjet1eta',
            'bjet2eta',
            'bjet3eta',
            'ljet1eta',
            'ljet2eta',
            'ljet3eta',
            'ljet4eta',
            'bjet1phi',
            'bjet2phi',
            'bjet3phi',
            'ljet1phi',
            'ljet2phi',
            'ljet3phi',
            'ljet4phi',
            'bjet1M',
            'bjet2M',
            'bjet3M',
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
            'lep1pT',
            'lep1eta',
            'lep1phi',
            'lep1M',
            'MET',
            'METsig',
            'METphi',
            'MET_Soft',
            'MET_Jet',
            'MET_Ele',
            'MET_Muon',
            'mT_METl',
            'dR_bb',
            'dphi_METl',
            'MT2_bb',
            'MT2_b1l1_b2',
            'MT2_b2l1_b1',
            'MT2_min',
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

variables = ['truth_top_min_dR',
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

if __name__ == '__main__':
    dataset = args.split
    with open(args.filelist) as f:
        i=0
        for line in f:
            filename = line.strip()
            print('reading : ',filename)
            sig=False
            if '_Signal_' in filename: sig=True
            with uproot.open({filename: "stop1L_NONE;1"}) as tree:
                branches = tree.arrays(Features)
                mask_i,inputs_i,targets_i,out_truth_info_i,met_i,signal_i = get_data(branches,dataset=dataset,sig=sig)
                if i==0:
                    mask = mask_i
                    inputs = inputs_i
                    targets = targets_i
                    out_truth_info = out_truth_info_i
                    met = met_i
                    signal = signal_i
                else:
                    mask = np.concatenate((mask,mask_i),axis=0)
                    inputs = merge(inputs,inputs_i)
                    targets = merge(targets,targets_i)
                    out_truth_info = merge(out_truth_info,out_truth_info_i)
                    met = merge(met,met_i)
                    signal = np.concatenate((signal,signal_i),axis=0)
                i+=1
            out_dir = f'{args.out_dir}/'
            if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
            out_f = out_dir + f'/spanet_inputs_{dataset}.h5'
            with h5py.File(out_f, 'w') as out_file: 
                classifications_group = out_file.create_group('CLASSIFICATIONS')
                event = classifications_group.create_group(f'EVENT')
                event.create_dataset('signal', data=signal, dtype='int64')
                match = np.maximum(out_truth_info['truth_topp_match'],out_truth_info['truth_topm_match'])
                match += 2
                only_sig_match = False
                if only_sig_match:
                    if sig == False:
                        match[match!=0] = -1
                event.create_dataset('match', data=match,dtype='int64')

                inputs_group = out_file.create_group('INPUTS')
                Momenta = inputs_group.create_group(f'Momenta')
                Momenta.create_dataset('MASK', data=mask, dtype='bool')
                Momenta.create_dataset('btag', data=inputs['btag'])
                Momenta.create_dataset('qtag', data=inputs['qtag'])
                Momenta.create_dataset('etag', data=inputs['etag'])
                Momenta.create_dataset('eta', data=inputs['eta'])
                Momenta.create_dataset('mass', data=inputs['M'])
                Momenta.create_dataset('phi', data=inputs['phi'])
                Momenta.create_dataset('pt', data=inputs['pT'])
               
                Met = inputs_group.create_group(f'Met')  
                Met.create_dataset('MET', data=met['MET'],dtype='float32')   
                Met.create_dataset('METsig', data=met['METsig'],dtype='float32')
                Met.create_dataset('METphi', data=met['METphi'],dtype='float32')
                Met.create_dataset('MET_Soft', data=met['MET_Soft'],dtype='float32')
                Met.create_dataset('MET_Jet', data=met['MET_Jet'],dtype='float32')
                Met.create_dataset('MET_Ele', data=met['MET_Ele'],dtype='float32')
                Met.create_dataset('MET_Muon', data=met['MET_Muon'],dtype='float32')
                Met.create_dataset('mT_METl', data=met['mT_METl'],dtype='float32')
                Met.create_dataset('dR_bb', data=met['dR_bb'],dtype='float32')
                Met.create_dataset('dphi_METl', data=met['dphi_METl'],dtype='float32')
                Met.create_dataset('MT2_bb', data=met['MT2_bb'],dtype='float32')
                Met.create_dataset('MT2_b1l1_b2', data=met['MT2_b1l1_b2'],dtype='float32')
                Met.create_dataset('MT2_b2l1_b1', data=met['MT2_b2l1_b1'],dtype='float32')
                Met.create_dataset('MT2_min', data=met['MT2_min'],dtype='float32') 
                
                targets_group = out_file.create_group('TARGETS')
                ht = targets_group.create_group(f'ht')
                ht.create_dataset('b', data=targets['htb'],dtype='int64')
                ht.create_dataset('q1', data=targets['q1'],dtype='int64')
                ht.create_dataset('q2', data=targets['q2'],dtype='int64')
                lt = targets_group.create_group(f'lt')
                lt.create_dataset('b', data=targets['ltb'],dtype='int64')
                lt.create_dataset('l', data=targets['ltl'],dtype='int64')

                regressions_group = out_file.create_group('REGRESSIONS')
                regression = regressions_group.create_group(f'EVENT')

                truth_info_group = out_file.create_group('truth_info')
                for info in out_truth_info.keys():
                    truth_info_group.create_dataset(info, data=out_truth_info[info])

    #with open(args.filelist) as f:
    #    for line in f:
    #        filename = line.strip()
    #        print('reading : ',filename)
    #        with uproot.open({filename: "stop1L_NONE;1"}) as tree:
    #            out_dir = f'{args.out_dir}_{dataset}/'
    #            branches = tree.arrays(Features)
    #            mask,inputs,targets,out_truth_info = get_data(branches,dataset=dataset)
    #            if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    #            data_index = filename.index("/MC")
    #            out_dir = out_dir + (filename[data_index:]).replace(".root",f"_{dataset}.h5")
    #            out_index = out_dir.index("/mc2") 
    #            if (not os.path.exists(out_dir[:out_index])): os.system(f'mkdir {out_dir[:out_index]}')
    #            with h5py.File(out_dir, 'w') as out_file: 
    #                inputs_group = out_file.create_group('INPUTS')
    #                source = inputs_group.create_group(f'Source')
    #                source.create_dataset('MASK', data=mask)
    #                source.create_dataset('btag', data=inputs['btag'])
    #                source.create_dataset('eta', data=inputs['eta'])
    #                source.create_dataset('mass', data=inputs['M'])
    #                source.create_dataset('phi', data=inputs['phi'])
    #                source.create_dataset('pt', data=inputs['pT'])
    #                targets_group = out_file.create_group('TARGETS')
    #                ht = targets_group.create_group(f'ht')
    #                ht.create_dataset('b', data=targets['htb'],dtype='i4')
    #                ht.create_dataset('q1', data=targets['q1'],dtype='i4')
    #                ht.create_dataset('q2', data=targets['q2'],dtype='i4')
    #                lt = targets_group.create_group(f'lt')
    #                lt.create_dataset('b', data=targets['ltb'],dtype='i4')