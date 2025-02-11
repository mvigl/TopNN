import numpy as np
import uproot
import awkward as ak
import argparse
import h5py
import os
import yaml

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='data/root/list_one.txt')
parser.add_argument('--split', help='train,test,val,even,odd',default='odd')
parser.add_argument('--out_dir', help='out_dir',default='SPANet_multi_class')
parser.add_argument('--combine',  action='store_true', help='combine', default=True)
parser.add_argument('--bkg_targets',  action='store_true', help='bkg_targets', default=True)
parser.add_argument('--massgrid', help='massgrid',default='/raven/u/mvigl/Stop/TopNN/data/stop_masses.yaml')
parser.add_argument('--sigsub', type=float, help='sigsub',default=1.)
args = parser.parse_args()

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
            #'ljet5pT',
            #'ljet6pT',
            'largejet1pT',
            'largejet2pT',
            #'largejet3pT',
            'bjet1eta',
            'bjet2eta',
            'bjet3eta',
            'ljet1eta',
            'ljet2eta',
            'ljet3eta',
            'ljet4eta',
            #'ljet5eta',
            #'ljet6eta',
            'largejet1eta',
            'largejet2eta',
            #'largejet3eta',
            'bjet1phi',
            'bjet2phi',
            'bjet3phi',
            'ljet1phi',
            'ljet2phi',
            'ljet3phi',
            'ljet4phi',
            #'ljet5phi',
            #'ljet6phi',
            'largejet1phi',
            'largejet2phi',
            #'largejet3phi',
            'bjet1M',
            'bjet2M',
            'bjet3M',
            'ljet1M',
            'ljet2M',
            'ljet3M',
            'ljet4M',
            #'ljet5M',
            #'ljet6M',
            'largejet1M',
            'largejet2M',
            #'largejet3M',
            'bjet1score',
            'bjet2score',
            'bjet3score',
            'ljet1score',
            'ljet2score',
            'ljet3score',
            'ljet4score',
            #'ljet5score',
            #'ljet6score',
            'largejet1toptagged',
            'largejet2toptagged',
            #'largejet3toptagged',
            'largejet1wtagged',
            'largejet2wtagged',
            #'largejet3wtagged',
            'largejet1ztagged',
            'largejet2ztagged',
            #'largejet3ztagged',
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
            'WeightEventsbTag',
            'WeightEventselSF',
            'WeightEventsJVT',
            'WeightEventsmuSF',
            'WeightEventsPU',
            'WeightEventsSF_global',
            'WeightEvents_trigger_ele_single',
            'WeightEvents_trigger_mu_single',
            'xsec',
            'WeightLumi',
            'lep1pT',
            'lep1eta',
            'lep1phi',
            'lep1M',
            'lepflav1',
            'MET',
            'METsig',
            'METphi',
            #'MET_Soft',
            #'MET_Jet',
            #'MET_Ele',
            #'MET_Muon',
            'mT_METl',
            'dR_bb',
            'dphi_METl',
            'MT2_bb',
            'MT2_b1l1_b2',
            'MT2_b2l1_b1',
            'MT2_min',
            'HT',
            'nbjet',
            'nljet',
            'njet',
            'nlep',
            'nlargejet',
            'nVx',
            'RunNumber',
            ]

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

def idxs_to_var(branches,dataset,sig,bkg_targets=False,sigsub=1):

    # dummy filter, here is all events
    filter = (ak.Array(branches['multiplets'])[:,0,-1] > -100)
    rnum_array = (ak.Array(branches['RunNumber'])[:])

    # no good had-top triplet/doublet
    not_matched =  (ak.Array(branches['multiplets'])[filter,0,-1]!=1)
    # no lep-top
    not_matched_l =  (((ak.Array(branches['truth_topp_match'])[filter].to_numpy()==-1)+(ak.Array(branches['truth_topm_match'])[filter].to_numpy()==-1))==0)
    
    length = np.sum(filter)
    vars=['btag','qtag','etag','bscore','Larget','LargeZ','LargeW','eta','M','phi','pT']
    inputs = {}
    # fill the 4-momenta info
    # will store as (bjet1,bjet2,ljet1,ljet2,ljet3,ljet4,lep,bjet3,0,0) - important for filling the idxs of reconstruction targets 
    # can add more jets in the future
    for var in vars:
        if var == 'btag':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,0] += 1
            inputs[var][:,1] += 1
            #inputs[var][:,7] += 1
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        elif var == 'qtag':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,2] += 1
            inputs[var][:,3] += 1
            inputs[var][:,4] += 1
            inputs[var][:,5] += 1 
            #inputs[var][:,8] += 1 
            #inputs[var][:,9] += 1  
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        elif var == 'Larget':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,7] += ak.Array(branches['largejet1toptagged'][filter]).to_numpy()
            inputs[var][:,8] += ak.Array(branches['largejet2toptagged'][filter]).to_numpy()
            #inputs[var][:,12] += ak.Array(branches['largejet3toptagged'][filter]).to_numpy()
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        elif var == 'LargeZ':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,7] += ak.Array(branches['largejet1ztagged'][filter]).to_numpy()
            inputs[var][:,8] += ak.Array(branches['largejet2ztagged'][filter]).to_numpy()
            #inputs[var][:,12] += ak.Array(branches['largejet3ztagged'][filter]).to_numpy()
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        elif var == 'LargeW':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,7] += ak.Array(branches['largejet1wtagged'][filter]).to_numpy()
            inputs[var][:,8] += ak.Array(branches['largejet2wtagged'][filter]).to_numpy()
            #inputs[var][:,12] += ak.Array(branches['largejet3wtagged'][filter]).to_numpy()
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)        
        elif var == 'bscore':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,0] += ak.Array(branches['bjet1score'][filter]).to_numpy()
            inputs[var][:,1] += ak.Array(branches['bjet2score'][filter]).to_numpy()
            #inputs[var][:,7] += ak.Array(branches['bjet3score'][filter]).to_numpy()
            inputs[var][:,2] += ak.Array(branches['ljet1score'][filter]).to_numpy()
            inputs[var][:,3] += ak.Array(branches['ljet2score'][filter]).to_numpy()
            inputs[var][:,4] += ak.Array(branches['ljet3score'][filter]).to_numpy()
            inputs[var][:,5] += ak.Array(branches['ljet4score'][filter]).to_numpy()
            #inputs[var][:,8] += ak.Array(branches['ljet5score'][filter]).to_numpy()
            #inputs[var][:,9] += ak.Array(branches['ljet6score'][filter]).to_numpy()
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        elif var == 'etag':
            inputs[var] = np.zeros((length,9))
            inputs[var][:,6] += 1
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)        
        else:
            inputs[var] = np.zeros((length,9))
            inputs[var][:,0] += ak.Array(branches['bjet1'+var][filter]).to_numpy()
            inputs[var][:,1] += ak.Array(branches['bjet2'+var][filter]).to_numpy()
            inputs[var][:,2] += ak.Array(branches['ljet1'+var][filter]).to_numpy()
            inputs[var][:,3] += ak.Array(branches['ljet2'+var][filter]).to_numpy()
            inputs[var][:,4] += ak.Array(branches['ljet3'+var][filter]).to_numpy()
            inputs[var][:,5] += ak.Array(branches['ljet4'+var][filter]).to_numpy()
            inputs[var][:,6] += ak.Array(branches['lep1'+var][filter]).to_numpy()
            #inputs[var][:,7] += ak.Array(branches['bjet3'+var][filter]).to_numpy()
            #inputs[var][:,8] += ak.Array(branches['ljet5'+var][filter]).to_numpy()
            #inputs[var][:,9] += ak.Array(branches['ljet6'+var][filter]).to_numpy()
            inputs[var][:,7] += ak.Array(branches['largejet1'+var][filter]).to_numpy()
            inputs[var][:,8] += ak.Array(branches['largejet2'+var][filter]).to_numpy()
            #inputs[var][:,12] += ak.Array(branches['largejet3'+var][filter]).to_numpy()
            (inputs[var])[inputs[var]==-999]=0.
            #(inputs[var])[inputs[var]==-10]=0.
            inputs[var] = split_data(length,inputs[var],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
    mask = (inputs['pT']>0)
    inputs['btag'][mask==False]=0.
    inputs['qtag'][mask==False]=0.
    inputs['etag'][mask==False]=0.
    inputs['bscore'][mask==False]=0.
    inputs['Larget'][mask==False]=0.
    inputs['LargeZ'][mask==False]=0.
    inputs['LargeW'][mask==False]=0.

    # fill the reconstruction targets
    targets = {}
    # deafault is not reco targets for bkg samples (truth matching was devoleped for 1L so might not be well defined for bkg - to be checked)
    if ((sig==False) and (bkg_targets==False)):
        targets['htb'] = -np.ones((length))
        targets['q1'] = -np.ones((length))
        targets['q2'] = -np.ones((length))
        targets['ltb'] = -np.ones((length))
        targets['ltl'] = -np.ones((length))
        targets['htb'] = split_data(length,targets['htb'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['q1'] = split_data(length,targets['q1'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['q2'] = split_data(length,targets['q2'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['ltb'] = split_data(length,targets['ltb'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['ltl'] = split_data(length,targets['ltl'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
    else:
        targets['htb'] = -np.ones((length))
        targets['q1'] = -np.ones((length))
        targets['q2'] = -np.ones((length))
        targets['ltb'] = -np.ones((length))
        targets['ltl'] = np.ones((length))*6
        # mutiplets have this structure: (bjetidx,ljetidx,ljetidx), and the truth matched are always in the first column(s) - but we only need one had-top for 1L
        # if more had-top per event are needed then this code should change a bit to loop over the first 2 multiplets
        # bjetIdxs_saved is just [bjetidx1,bjetidx2]
        # ljetIdxs_saved is just [ljetidx1,ljetidx2,ljetidx3,ljetidx4]
        # so check which index is in the first multiplet and map it to the (bjet1,bjet2,ljet1,ljet2,ljet3,ljet4,lep,bjet3,0,0) input structure built before
        targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,0]))] = 0
        targets['htb'][(ak.Array(branches['multiplets'][filter,0,0])==ak.Array(branches['bjetIdxs_saved'][filter,1]))] = 1
        targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
        targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
        targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
        targets['q1'][(ak.Array(branches['multiplets'][filter,0,1])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5
        targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,0]))] = 2
        targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,1]))] = 3
        targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,2]))] = 4
        targets['q2'][(ak.Array(branches['multiplets'][filter,0,2])==ak.Array(branches['ljetIdxs_saved'][filter,3]))] = 5

        # restore -1 value for not matched multiplets
        targets['htb'][not_matched] = -1
        targets['q1'][not_matched] = -1
        targets['q2'][not_matched] = -1
        # assign the leading bjet (not already matched to had-top) to lep-top
        targets['ltb'] = targets['htb']+1
        targets['ltb'][targets['ltb']==2] = 0
        # restore -1 value for not matched lep-top
        targets['ltb'][not_matched_l] = -1
        targets['ltl'][not_matched_l] = -1
        (targets['ltb'])[not_matched] = -1 # if no had-top then don't reconstruct the lep-top (truth matching not well defined!!!!)
        (targets['ltl'])[not_matched] = -1 # if no had-top then don't reconstruct the lep-top (truth matching not well defined!!!!)

        targets['htb'] = split_data(length,targets['htb'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['q1'] = split_data(length,targets['q1'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['q2'] = split_data(length,targets['q2'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['ltb'] = split_data(length,targets['ltb'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        single_b = ((mask[:,0]*mask[:,1])==False)
        (targets['ltb'])[single_b*(targets['htb']!=-1)]=-1
        targets['ltl'] = split_data(length,targets['ltl'],rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
        targets['ltl'][(mask[:,6]==False)]=-1

    # fill the global features
    met = {
        'MET': split_data(length,branches['MET'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'METsig': split_data(length,branches['METsig'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'METphi': split_data(length,branches['METphi'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        #'MET_Soft': split_data(length,branches['MET_Soft'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        #'MET_Jet': split_data(length,branches['MET_Jet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        #'MET_Ele': split_data(length,branches['MET_Ele'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        #'MET_Muon': split_data(length,branches['MET_Muon'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'mT_METl': split_data(length,branches['mT_METl'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'dR_bb': split_data(length,branches['dR_bb'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'dphi_METl': split_data(length,branches['dphi_METl'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'MT2_bb': split_data(length,branches['MT2_bb'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'MT2_b1l1_b2': split_data(length,branches['MT2_b1l1_b2'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'MT2_b2l1_b1': split_data(length,branches['MT2_b2l1_b1'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'MT2_min': split_data(length,branches['MT2_min'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'HT': split_data(length,branches['HT'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nbjet': split_data(length,branches['nbjet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nljet': split_data(length,branches['nljet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nlargejet': split_data(length,branches['nlargejet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nVx': split_data(length,branches['nVx'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'lepflav1': split_data(length,branches['lepflav1'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
    }

    # fill some aux info
    truth_info = {
        'truth_top_min_dR': split_data(length,branches['truth_top_min_dR'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_top_min_dR_m': split_data(length,branches['truth_top_min_dR_m'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_top_min_dR_jj': split_data(length,branches['truth_top_min_dR_jj'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_top_min_dR_m_jj': split_data(length,branches['truth_top_min_dR_m_jj'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_topp_match': split_data(length,branches['truth_topp_match'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_topm_match': split_data(length,branches['truth_topm_match'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_topp_pt': split_data(length,branches['truth_topp_pt'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_topm_pt': split_data(length,branches['truth_topm_pt'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_Wp_pt': split_data(length,branches['truth_Wp_pt'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'truth_Wm_pt': split_data(length,branches['truth_Wm_pt'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEvents': split_data(length,branches['WeightEvents'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventsbTag': split_data(length,branches['WeightEventsbTag'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventselSF': split_data(length,branches['WeightEventselSF'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventsJVT': split_data(length,branches['WeightEventsJVT'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventsmuSF': split_data(length,branches['WeightEventsmuSF'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventsPU': split_data(length,branches['WeightEventsPU'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEventsSF_global': split_data(length,branches['WeightEventsSF_global'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEvents_trigger_ele_single': split_data(length,branches['WeightEvents_trigger_ele_single'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightEvents_trigger_mu_single': split_data(length,branches['WeightEvents_trigger_mu_single'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'xsec': split_data(length,branches['xsec'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'WeightLumi': split_data(length,branches['WeightLumi'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nbjet': split_data(length,branches['nbjet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nljet': split_data(length,branches['nljet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'njet': split_data(length,branches['njet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nlep': split_data(length,branches['nlep'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nlargejet': split_data(length,branches['nlargejet'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'nVx': split_data(length,branches['nVx'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'RunNumber': split_data(length,branches['RunNumber'][filter].to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub),
        'is_matched': split_data(length,(branches['multiplets'][filter,0,-1]==1).to_numpy(),rnum_array=rnum_array,dataset=dataset,sig=sig,sigsub=sigsub)
    }
    truth_info['training_weights'] = np.abs(truth_info['WeightEvents'])
    for w in ['WeightEventsbTag','WeightEventselSF','WeightEventsJVT','WeightEventsmuSF','WeightEventsPU','WeightEvents_trigger_ele_single','WeightEvents_trigger_mu_single','xsec','WeightLumi']:
        truth_info['training_weights']*=truth_info[w]
    return mask,inputs,targets,truth_info,met

def get_data(branches,massgrid,dataset='train',sig=True,number=3456,bkg_targets=False,sigsub=1):
    mask,inputs,targets,truth_info,met = idxs_to_var(branches,dataset,sig,bkg_targets,sigsub)
    if sig:
        signal = np.ones(len(mask))
    else:
        signal = np.zeros(len(mask))    
        # signal mass info
    with open(massgrid) as file:
        map = yaml.load(file, Loader=yaml.FullLoader)['samples'] 
    m1=(map[number])[0]
    m2=(map[number])[1]
    truth_info['M1'] = np.ones(len(mask))*m1
    truth_info['M2'] = np.ones(len(mask))*m2
    if m1<0:
        truth_info['p1']=truth_info['M1']
        truth_info['p2']=truth_info['M2']
    elif m1-m2<=350:
        truth_info['p1']=np.ones(len(mask))*0 
        truth_info['p2']=np.ones(len(mask))*0  
    elif m1-m2<=650:
        truth_info['p1']=np.ones(len(mask))*0 
        truth_info['p2']=np.ones(len(mask))*1  
    elif m1>=1250:
        truth_info['p1']=np.ones(len(mask))*1 
        truth_info['p2']=np.ones(len(mask))*1    
    else:
        truth_info['p1']=np.ones(len(mask))*1 
        truth_info['p2']=np.ones(len(mask))*0    

    multi_class = np.ones(len(mask))*((map[number])[2])
    
    return mask,inputs,targets,truth_info,met,signal,multi_class

def merge(d1,d2):
    merged_dict = {}
    for key in d1.keys():
        merged_dict[key] = np.concatenate((d1[key],d2[key]),axis=0)
    return merged_dict

def save_combined(args):
        with open(args.filelist) as f:
            i=0
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                number = filename[(filename.index("TeV.")+4):(filename.index(".stop1L"))]
                with open(args.massgrid) as g:
                    map = yaml.load(g, Loader=yaml.FullLoader)['samples'] 
                if number not in map.keys(): 
                    print('--- skipping sample ---')
                    continue
                sig=False
                if '_Signal_' in filename: sig=True
                mc_list=''
                if 'MCRun2_' in filename: mc_list = '/raven/u/mvigl/Stop/TopNN/data/input_list_mc20.yaml'
                else: mc_list = '/raven/u/mvigl/Stop/TopNN/data/input_list_mc23.yaml'
                with open(f'{mc_list}') as l:
                    map = yaml.load(l, Loader=yaml.FullLoader)['samples']
                    sample=(map[number])
                try:
                    with uproot.open(f'{filename}') as tree:
                        if len(tree.keys())==0:
                            print('file empty \n')
                            continue
                except Exception as e:
                    print(f'{e} \n')
                    print(f'Cannot open file')
                    continue    
                with uproot.open({filename: f'{sample}_NONE'}) as tree: #stop1L_NONE
                    branches = tree.arrays(Features)
                    mask_i,inputs_i,targets_i,out_truth_info_i,met_i,signal_i,multi_class_i = get_data(branches,args.massgrid,dataset=dataset,sig=sig,number=number,bkg_targets=args.bkg_targets,sigsub=args.sigsub)
                    if i==0:
                        mask = mask_i
                        inputs = inputs_i
                        targets = targets_i
                        out_truth_info = out_truth_info_i
                        met = met_i
                        signal = signal_i
                        multi_class = multi_class_i
                    else:
                        mask = np.concatenate((mask,mask_i),axis=0)
                        inputs = merge(inputs,inputs_i)
                        targets = merge(targets,targets_i)
                        out_truth_info = merge(out_truth_info,out_truth_info_i)
                        met = merge(met,met_i)
                        signal = np.concatenate((signal,signal_i),axis=0)
                        multi_class = np.concatenate((multi_class,multi_class_i),axis=0)
                    i+=1
            out_dir = f'{args.out_dir}/'
            if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
            if args.bkg_targets: out_f = out_dir + f'/spanet_inputs_{dataset}_{args.sigsub}.h5'
            else: out_f = out_dir + f'/spanet_inputs_{dataset}_{args.sigsub}_no_bkg_reco.h5'
            with h5py.File(out_f, 'w') as out_file: 
                classifications_group = out_file.create_group('CLASSIFICATIONS')
                event = classifications_group.create_group(f'EVENT')
                event.create_dataset('signal', data=signal, dtype='int64')
                event.create_dataset('class', data=multi_class, dtype='int64')
                match_p = out_truth_info['truth_topp_match']
                match_m = out_truth_info['truth_topm_match']
                match_p += 1
                match_m += 1
                # take the highest reconstructuble had-top match as target (if more than one had-top cannot recover 2d info at the moment)
                match = np.maximum(match_p,match_m)
                match[match==-1]=0 
                event.create_dataset('match', data=match,dtype='int64')

                inputs_group = out_file.create_group('INPUTS')
                Momenta = inputs_group.create_group(f'Momenta')
                Momenta.create_dataset('MASK', data=mask, dtype='bool')
                Momenta.create_dataset('btag', data=inputs['btag'])
                Momenta.create_dataset('qtag', data=inputs['qtag'])
                Momenta.create_dataset('etag', data=inputs['etag'])
                Momenta.create_dataset('bscore', data=inputs['bscore'])
                Momenta.create_dataset('Larget', data=inputs['Larget'])
                Momenta.create_dataset('LargeZ', data=inputs['LargeZ'])
                Momenta.create_dataset('LargeW', data=inputs['LargeW'])
                Momenta.create_dataset('eta', data=inputs['eta'])
                Momenta.create_dataset('mass', data=inputs['M'])
                Momenta.create_dataset('phi', data=inputs['phi'])
                Momenta.create_dataset('pt', data=inputs['pT'])

                Met = inputs_group.create_group(f'Met')  
                Met.create_dataset('MET', data=met['MET'],dtype='float32')   
                Met.create_dataset('METsig', data=met['METsig'],dtype='float32')
                Met.create_dataset('METphi', data=met['METphi'],dtype='float32')
                #Met.create_dataset('MET_Soft', data=met['MET_Soft'],dtype='float32')
                #Met.create_dataset('MET_Jet', data=met['MET_Jet'],dtype='float32')
                #Met.create_dataset('MET_Ele', data=met['MET_Ele'],dtype='float32')
                #Met.create_dataset('MET_Muon', data=met['MET_Muon'],dtype='float32')
                Met.create_dataset('mT_METl', data=met['mT_METl'],dtype='float32')
                Met.create_dataset('dR_bb', data=met['dR_bb'],dtype='float32')
                Met.create_dataset('dphi_METl', data=met['dphi_METl'],dtype='float32')
                Met.create_dataset('MT2_bb', data=met['MT2_bb'],dtype='float32')
                Met.create_dataset('MT2_b1l1_b2', data=met['MT2_b1l1_b2'],dtype='float32')
                Met.create_dataset('MT2_b2l1_b1', data=met['MT2_b2l1_b1'],dtype='float32')
                Met.create_dataset('MT2_min', data=met['MT2_min'],dtype='float32') 
                Met.create_dataset('HT', data=met['HT'],dtype='float32') 
                Met.create_dataset('nbjet', data=met['nbjet'],dtype='int64') 
                Met.create_dataset('nljet', data=met['nljet'],dtype='int64') 
                Met.create_dataset('nlargejet', data=met['nlargejet'],dtype='int64') 
                Met.create_dataset('nVx', data=met['nVx'],dtype='int64')
                Met.create_dataset('lepflav1', data=met['lepflav1'],dtype='int64')
                Met.create_dataset('M1', data=out_truth_info['M1'],dtype='float32')   
                Met.create_dataset('M2', data=out_truth_info['M2'],dtype='float32') 
                Met.create_dataset('p1', data=out_truth_info['p1'],dtype='float32')   
                Met.create_dataset('p2', data=out_truth_info['p2'],dtype='float32')   

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

                weights_group = out_file.create_group('WEIGHTS')
                weights = weights_group.create_group(f'EVENT')
                weights.create_dataset('event_weights',data=out_truth_info['training_weights'],dtype='float32')

                truth_info_group = out_file.create_group('truth_info')
                for info in out_truth_info.keys():
                    truth_info_group.create_dataset(info, data=out_truth_info[info])

def save_single(args):
        with open(args.filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                sig=False
                if '_Signal_' in filename: sig=True
                number = number = filename[(filename.index("TeV.")+4):(filename.index(".stop1L"))]
                sub_dir = filename[(filename.index("/MC")):(filename.index("/mc2"))]
                if args.bkg_targets: out_f = filename[(filename.index("/mc2")):].replace(".root",f"_{dataset}_{args.sigsub}.h5")
                else: out_f = filename[(filename.index("/mc2")):].replace(".root",f"_{dataset}_{args.sigsub}_no_bkg_reco.h5")
                #skip unkown samples:
                with open(args.massgrid) as g:
                    map = yaml.load(g, Loader=yaml.FullLoader)['samples'] 
                if number not in map.keys(): 
                    print('--- skipping sample ---')
                    continue
                mc_list=''
                if 'MCRun2_' in filename: mc_list = '/raven/u/mvigl/Stop/TopNN/data/input_list_mc20.yaml'
                else: mc_list = '/raven/u/mvigl/Stop/TopNN/data/input_list_mc23.yaml'
                with open(f'{mc_list}') as l:
                    map = yaml.load(l, Loader=yaml.FullLoader)['samples']
                    sample=(map[number])
                try:
                    with uproot.open(f'{filename}') as tree:
                        if len(tree.keys())==0:
                            print('file empty \n')
                            continue
                except Exception as e:
                    print(f'{e} \n')
                    print(f'Cannot open file')
                    continue
                with uproot.open({filename: f'{sample}_NONE'}) as tree: #HistFitterTree_NONE
                    branches = tree.arrays(Features)
                    

                    mask,inputs,targets,out_truth_info,met,signal,multi_class = get_data(branches,args.massgrid,dataset=dataset,sig=sig,number=number,bkg_targets=args.bkg_targets,sigsub=args.sigsub)
                    out_dir = f'{args.out_dir}/'
                    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
                    sub_dir = f'{out_dir}{sub_dir}'
                    if (not os.path.exists(sub_dir)): os.system(f'mkdir {sub_dir}')
                    out_f = f'{sub_dir}{out_f}'
                    with h5py.File(out_f, 'w') as out_file: 
                        classifications_group = out_file.create_group('CLASSIFICATIONS')
                        event = classifications_group.create_group(f'EVENT')
                        event.create_dataset('signal', data=signal, dtype='int64')
                        event.create_dataset('class', data=multi_class, dtype='int64')
                        match_p = out_truth_info['truth_topp_match']
                        match_m = out_truth_info['truth_topm_match']
                        match_p += 1
                        match_m += 1
                        # take the highest reconstructuble had-top match as target (if more than one had-top cannot recover 2d info at the moment)
                        match = np.maximum(match_p,match_m)
                        match[match==-1]=0 
                        event.create_dataset('match', data=match,dtype='int64')

                        inputs_group = out_file.create_group('INPUTS')
                        Momenta = inputs_group.create_group(f'Momenta')
                        Momenta.create_dataset('MASK', data=mask, dtype='bool')
                        Momenta.create_dataset('btag', data=inputs['btag'])
                        Momenta.create_dataset('qtag', data=inputs['qtag'])
                        Momenta.create_dataset('etag', data=inputs['etag'])
                        Momenta.create_dataset('bscore', data=inputs['bscore'])
                        Momenta.create_dataset('Larget', data=inputs['Larget'])
                        Momenta.create_dataset('LargeZ', data=inputs['LargeZ'])
                        Momenta.create_dataset('LargeW', data=inputs['LargeW'])
                        Momenta.create_dataset('eta', data=inputs['eta'])
                        Momenta.create_dataset('mass', data=inputs['M'])
                        Momenta.create_dataset('phi', data=inputs['phi'])
                        Momenta.create_dataset('pt', data=inputs['pT'])

                        Met = inputs_group.create_group(f'Met')  
                        Met.create_dataset('MET', data=met['MET'],dtype='float32')   
                        Met.create_dataset('METsig', data=met['METsig'],dtype='float32')
                        Met.create_dataset('METphi', data=met['METphi'],dtype='float32')
                        #Met.create_dataset('MET_Soft', data=met['MET_Soft'],dtype='float32')
                        #Met.create_dataset('MET_Jet', data=met['MET_Jet'],dtype='float32')
                        #Met.create_dataset('MET_Ele', data=met['MET_Ele'],dtype='float32')
                        #Met.create_dataset('MET_Muon', data=met['MET_Muon'],dtype='float32')
                        Met.create_dataset('mT_METl', data=met['mT_METl'],dtype='float32')
                        Met.create_dataset('dR_bb', data=met['dR_bb'],dtype='float32')
                        Met.create_dataset('dphi_METl', data=met['dphi_METl'],dtype='float32')
                        Met.create_dataset('MT2_bb', data=met['MT2_bb'],dtype='float32')
                        Met.create_dataset('MT2_b1l1_b2', data=met['MT2_b1l1_b2'],dtype='float32')
                        Met.create_dataset('MT2_b2l1_b1', data=met['MT2_b2l1_b1'],dtype='float32')
                        Met.create_dataset('MT2_min', data=met['MT2_min'],dtype='float32') 
                        Met.create_dataset('HT', data=met['HT'],dtype='float32') 
                        Met.create_dataset('nbjet', data=met['nbjet'],dtype='int64') 
                        Met.create_dataset('nljet', data=met['nljet'],dtype='int64') 
                        Met.create_dataset('nlargejet', data=met['nlargejet'],dtype='int64') 
                        Met.create_dataset('nVx', data=met['nVx'],dtype='int64')
                        Met.create_dataset('lepflav1', data=met['lepflav1'],dtype='int64')
                        Met.create_dataset('M1', data=out_truth_info['M1'],dtype='float32')   
                        Met.create_dataset('M2', data=out_truth_info['M2'],dtype='float32')
                        Met.create_dataset('p1', data=out_truth_info['p1'],dtype='float32')   
                        Met.create_dataset('p2', data=out_truth_info['p2'],dtype='float32')   

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

                        weights_group = out_file.create_group('WEIGHTS')
                        weights = weights_group.create_group(f'EVENT')
                        weights.create_dataset('event_weights',data=out_truth_info['training_weights'],dtype='float32')

                        truth_info_group = out_file.create_group('truth_info')
                        for info in out_truth_info.keys():
                            truth_info_group.create_dataset(info, data=out_truth_info[info])                        
                        
if __name__ == '__main__':
    dataset = args.split

    if args.combine : 
        # loop over all list of files and save one .h5 file
        save_combined(args)
    else: 
        # loop over all list of files and save multiple .h5 file
        save_single(args)