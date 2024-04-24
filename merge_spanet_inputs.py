import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os



def read_file(file):    
    with h5py.File(file,'r') as h5fw :    

        signal = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
        match = h5fw['CLASSIFICATIONS']['EVENT']['match'][:]
        MASK = h5fw['INPUTS']['Momenta']['MASK'][:]
        btag = h5fw['INPUTS']['Momenta']['btag'][:]
        qtag = h5fw['INPUTS']['Momenta']['qtag'][:]
        etag = h5fw['INPUTS']['Momenta']['etag'][:]
        eta = h5fw['INPUTS']['Momenta']['eta'][:]
        mass = h5fw['INPUTS']['Momenta']['mass'][:]
        phi = h5fw['INPUTS']['Momenta']['phi'][:]
        pt = h5fw['INPUTS']['Momenta']['pt'][:]
        MET = h5fw['INPUTS']['Met']['MET'][:]
        METsig = h5fw['INPUTS']['Met']['METsig'][:]
        METphi = h5fw['INPUTS']['Met']['METphi'][:]
        MET_Soft = h5fw['INPUTS']['Met']['MET_Soft'][:]
        MET_Jet = h5fw['INPUTS']['Met']['MET_Jet'][:]
        MET_Ele = h5fw['INPUTS']['Met']['MET_Ele'][:]
        MET_Muon = h5fw['INPUTS']['Met']['MET_Muon'][:]
        mT_METl = h5fw['INPUTS']['Met']['mT_METl'][:]
        dR_bb = h5fw['INPUTS']['Met']['dR_bb'][:]
        dphi_METl = h5fw['INPUTS']['Met']['dphi_METl'][:]
        MT2_bb = h5fw['INPUTS']['Met']['MT2_bb'][:]
        MT2_b1l1_b2 = h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:]
        MT2_b2l1_b1 = h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:]
        MT2_min = h5fw['INPUTS']['Met']['MT2_min'][:]               
        htb = h5fw['TARGETS']['ht']['b'][:]
        q1 = h5fw['TARGETS']['ht']['q1'][:]
        q2 = h5fw['TARGETS']['ht']['q2'][:]
        ltb = h5fw['TARGETS']['lt']['ltb'][:]
        ltl = h5fw['TARGETS']['lt']['ltl'][:]
        truth_info = {}
        for info in h5fw['truth_info'].keys():
            truth_info[info] = h5fw['truth_info'][info]

        return  {'signal': signal ,
                'match': match ,
                'MASK': MASK ,
                'btag': btag ,
                'qtag': qtag ,
                'etag': etag ,
                'eta': eta ,
                'mass': mass ,
                'phi': phi ,
                'pt': pt ,
                'MET': MET ,
                'METsi': METsig,
                'METphi': METphi ,
                'MET_Sof': MET_Soft,
                'MET_Je': MET_Jet,
                'MET_El': MET_Ele,
                'MET_Muon': MET_Muon ,
                'mT_METl': mT_METl ,
                'dR_bb': dR_bb ,
                'dphi_METl': dphi_METl ,
                'MT2_bb': MT2_bb ,
                'MT2_b1l1_b2': MT2_b1l1_b2 ,
                'MT2_b2l1_b1': MT2_b2l1_b1 ,
                'MT2_min': MT2_min ,
                'htb': htb ,
                'q1': q1 ,
                'q2': q2 ,
                'ltb': ltb ,
                'ltl': ltl ,
                'truth_info': truth_info}

i=0
filelist = ['/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_FS/spanet_inputs_train.h5',
            '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_mc20a/spanet_inputs_train.h5',
            '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_mc20d_1/spanet_inputs_train.h5',
            '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_mc20d_2/spanet_inputs_train.h5',
            '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_mc20e/spanet_inputs_train.h5',
            '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_mc23/spanet_inputs_train.h5',
]
for file in filelist:
    merged = {}
    data = read_file(file)
    for key in data:
        if i==0 : merged[key] = data[key]
        else:
            if key=='truth_info':
                for sub_key in merged[key].keys():
                    merged[key][sub_key] = data[key][sub_key]
            else:
                merged[key] = np.concatenate((merged[key],data[key]),axis=0)

out_f = '/raven/u/mvigl/Stop/run/pre/H5_ete_spanet_stop_all/spanet_inputs_train.h5'
with h5py.File(out_f, 'w') as out_file: 
    classifications_group = out_file.create_group('CLASSIFICATIONS')
    event = classifications_group.create_group(f'EVENT')
    event.create_dataset('signal', data=merged['signal'], dtype='int64')
    event.create_dataset('match', data=merged['match'],dtype='int64')
    inputs_group = out_file.create_group('INPUTS')
    Momenta = inputs_group.create_group(f'Momenta')
    Momenta.create_dataset('MASK', data=merged['MASK'], dtype='bool')
    Momenta.create_dataset('btag', data=merged['btag'])
    Momenta.create_dataset('qtag', data=merged['qtag'])
    Momenta.create_dataset('etag', data=merged['etag'])
    Momenta.create_dataset('eta', data=merged['eta'])
    Momenta.create_dataset('mass', data=merged['mass'])
    Momenta.create_dataset('phi', data=merged['phi'])
    Momenta.create_dataset('pt', data=merged['pt'])
    Met = inputs_group.create_group(f'Met')  
    Met.create_dataset('MET', data=merged['match'],dtype='float32')   
    Met.create_dataset('METsig', data=merged['METsig'],dtype='float32')
    Met.create_dataset('METphi', data=merged['METphi'],dtype='float32')
    Met.create_dataset('MET_Soft', data=merged['MET_Soft'],dtype='float32')
    Met.create_dataset('MET_Jet', data=merged['MET_Jet'],dtype='float32')
    Met.create_dataset('MET_Ele', data=merged['MET_Ele'],dtype='float32')
    Met.create_dataset('MET_Muon', data=merged['MET_Muon'],dtype='float32')
    Met.create_dataset('mT_METl', data=merged['mT_METl'],dtype='float32')
    Met.create_dataset('dR_bb', data=merged['dR_bb'],dtype='float32')
    Met.create_dataset('dphi_METl', data=merged['dphi_METl'],dtype='float32')
    Met.create_dataset('MT2_bb', data=merged['MT2_bb'],dtype='float32')
    Met.create_dataset('MT2_b1l1_b2', data=merged['MT2_b1l1_b2'],dtype='float32')
    Met.create_dataset('MT2_b2l1_b1', data=merged['MT2_b2l1_b1'],dtype='float32')
    Met.create_dataset('MT2_min', data=merged['MT2_min'],dtype='float32')      
    targets_group = out_file.create_group('TARGETS')
    ht = targets_group.create_group(f'ht')
    ht.create_dataset('b', data=merged['htb'],dtype='int64')
    ht.create_dataset('q1', data=merged['q1'],dtype='int64')
    ht.create_dataset('q2', data=merged['q2'],dtype='int64')
    lt = targets_group.create_group(f'lt')
    lt.create_dataset('b', data=merged['ltb'],dtype='int64')
    lt.create_dataset('l', data=merged['ltl'],dtype='int64')
    regressions_group = out_file.create_group('REGRESSIONS')
    regression = regressions_group.create_group(f'EVENT')
    truth_info_group = out_file.create_group('truth_info')
    for info in merged['truth_info'].keys():
        truth_info_group.create_dataset(info, data=merged['truth_info'][info])



    
