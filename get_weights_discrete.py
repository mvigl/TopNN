import h5py
import os
import numpy as np
import pickle
from sklearn.metrics import roc_curve,auc
import matplotlib
import matplotlib.pyplot as plt

masses = ['500_1','500_100','500_200','500_300',
          '600_1','600_100','600_200','600_300','600_400','650_450',
          '700_1','700_100','700_200','700_300','700_400','700_500','750_550',
          '800_1','800_100','800_200','800_300','800_400','800_500','800_600','850_650',
          '900_1','900_100','900_200','900_300','900_400','900_500','900_600','900_700',
          '1000_1','1000_100','1000_200','1000_300','1000_400','1000_500','1000_600','1000_700','1000_800',
          '1100_1','1100_100','1100_200','1100_300','1100_400','1100_500','1100_600','1100_700','1100_800',
          '1200_1','1200_100','1200_200','1200_300','1200_400','1200_500','1200_600','1200_700','1200_800',
          '1300_1','1300_100','1300_200','1300_300','1300_400','1300_500','1300_600','1300_700','1300_800',
          '1400_1','1400_100','1400_200','1400_300','1400_400','1400_500','1400_600','1400_700','1400_800',
          '1500_1','1500_100','1500_200','1500_300','1500_400','1500_500','1500_600','1500_700','1500_800',
          '1600_1','1600_100','1600_200','1600_300','1600_400','1600_500','1600_600','1600_700','1600_800'
          ]

masses_slice = [[500,1],[500,100],[500,200],[500,300],
                [600,1],[600,100],[600,200],[600,300],[600,400],[650,450],
                [700,1],[700,100],[700,200],[700,300],[700,400],[700,500],[750,550],
                [800,1],[800,100],[800,200],[800,300],[800,400],[800,500],[800,600],[850,650],
                [900,1],[900,100],[900,200],[900,300],[900,400],[900,500],[900,600],[900,700],
                [1000,1],[1000,100],[1000,200],[1000,300],[1000,400],[1000,500],[1000,600],[1000,700],[1000,800],
                [1100,1],[1100,100],[1100,200],[1100,300],[1100,400],[1100,500],[1100,600],[1100,700],[1100,800],
                [1200,1],[1200,100],[1200,200],[1200,300],[1200,400],[1200,500],[1200,600],[1200,700],[1200,800],
                [1300,1],[1300,100],[1300,200],[1300,300],[1300,400],[1300,500],[1300,600],[1300,700],[1300,800],
                [1400,1],[1400,100],[1400,200],[1400,300],[1400,400],[1400,500],[1400,600],[1400,700],[1400,800],
                [1500,1],[1500,100],[1500,200],[1500,300],[1500,400],[1500,500],[1500,600],[1500,700],[1500,800],
                [1600,1],[1600,100],[1600,200],[1600,300],[1600,400],[1600,500],[1600,600],[1600,700],[1600,800]]

params = ['0_0','0_1','1_0','1_1']

params_slice = [[0,0],[0,1],[1,0],[1,1]]

with h5py.File("/raven/u/mvigl/Stop/run/pre/SPANet_multi_class/spanet_inputs_odd_1.0.h5",'r') as h5fw :  
    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
    event_weights = h5fw['truth_info']['training_weights'][:]
    multi_class = h5fw['CLASSIFICATIONS']['EVENT']['class'][:]
    print(multi_class)
    M12 = np.array([
                    h5fw['INPUTS']['Met']['M1'][:],
                    h5fw['INPUTS']['Met']['M2'][:],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    p12 = np.array([
                    h5fw['INPUTS']['Met']['p1'][:],
                    h5fw['INPUTS']['Met']['p2'][:],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    
    sig = np.sum(event_weights[(y==1)])
    for i,m in enumerate(masses):
        idxs = ((np.sum(M12==masses_slice[i],axis=-1)==2)).reshape(-1)
        sample = np.sum(event_weights[idxs])
        event_weights[idxs]*=(sig/sample)
    
    #sig = np.sum(event_weights[(y==1)])    
    #for i,m in enumerate(params):
    #    idxs = ((np.sum(p12==params_slice[i],axis=-1)==2)).reshape(-1)
    #    sample = np.sum(event_weights[idxs])
    #    event_weights[idxs]*=(sig/sample)    
    
    #tot = np.sum(event_weights)/5
    #for c in range(5):
    #    event_weights[multi_class==c]*=tot/np.sum(event_weights[multi_class==c]) 
    #event_weights=event_weights/np.mean(event_weights)
    sig = np.sum(event_weights[(y==1)])
    bkg = np.sum(event_weights[(y==0)]) 
    event_weights[(y==0)]*=sig/bkg
    event_weights=event_weights/np.mean(event_weights)
    with h5py.File("/raven/u/mvigl/Stop/run/pre/SPANet_multi_class/spanet_inputs_odd_weighted.h5", 'w') as out_file: 
        classifications_group = out_file.create_group('CLASSIFICATIONS')
        event = classifications_group.create_group(f'EVENT')
        event.create_dataset('signal', data=h5fw['CLASSIFICATIONS']['EVENT']['signal'], dtype='int64')
        event.create_dataset('class', data=h5fw['CLASSIFICATIONS']['EVENT']['class'], dtype='int64')
        event.create_dataset('match', data=h5fw['CLASSIFICATIONS']['EVENT']['match'],dtype='int64')
        inputs_group = out_file.create_group('INPUTS')
        Momenta = inputs_group.create_group(f'Momenta')
        Momenta.create_dataset('MASK', data=h5fw['INPUTS']['Momenta']['MASK'], dtype='bool')
        Momenta.create_dataset('btag', data=h5fw['INPUTS']['Momenta']['btag'])
        Momenta.create_dataset('qtag', data=h5fw['INPUTS']['Momenta']['qtag'])
        Momenta.create_dataset('etag', data=h5fw['INPUTS']['Momenta']['etag'])
        Momenta.create_dataset('bscore', data=h5fw['INPUTS']['Momenta']['bscore'])
        Momenta.create_dataset('Larget', data=h5fw['INPUTS']['Momenta']['Larget'])
        Momenta.create_dataset('LargeZ', data=h5fw['INPUTS']['Momenta']['LargeZ'])
        Momenta.create_dataset('LargeW', data=h5fw['INPUTS']['Momenta']['LargeW'])
        Momenta.create_dataset('eta', data=h5fw['INPUTS']['Momenta']['eta'])
        Momenta.create_dataset('mass', data=h5fw['INPUTS']['Momenta']['mass'])
        Momenta.create_dataset('phi', data=h5fw['INPUTS']['Momenta']['phi'])
        Momenta.create_dataset('pt', data=h5fw['INPUTS']['Momenta']['pt'])
        Met = inputs_group.create_group(f'Met')  
        Met.create_dataset('MET', data=h5fw['INPUTS']['Met']['MET'],dtype='float32')   
        Met.create_dataset('METsig', data=h5fw['INPUTS']['Met']['METsig'],dtype='float32')
        Met.create_dataset('METphi', data=h5fw['INPUTS']['Met']['METphi'],dtype='float32')
        #Met.create_dataset('MET_Soft', data=h5fw['INPUTS']['Met']['MET_Soft'],dtype='float32')
        #Met.create_dataset('MET_Jet', data=h5fw['INPUTS']['Met']['MET_Jet'],dtype='float32')
        #Met.create_dataset('MET_Ele', data=h5fw['INPUTS']['Met']['MET_Ele'],dtype='float32')
        #Met.create_dataset('MET_Muon', data=h5fw['INPUTS']['Met']['MET_Muon'],dtype='float32')
        Met.create_dataset('mT_METl', data=h5fw['INPUTS']['Met']['mT_METl'],dtype='float32')
        Met.create_dataset('dR_bb', data=h5fw['INPUTS']['Met']['dR_bb'],dtype='float32')
        Met.create_dataset('dphi_METl', data=h5fw['INPUTS']['Met']['dphi_METl'],dtype='float32')
        Met.create_dataset('MT2_bb', data=h5fw['INPUTS']['Met']['MT2_bb'],dtype='float32')
        Met.create_dataset('MT2_b1l1_b2', data=h5fw['INPUTS']['Met']['MT2_b1l1_b2'],dtype='float32')
        Met.create_dataset('MT2_b2l1_b1', data=h5fw['INPUTS']['Met']['MT2_b2l1_b1'],dtype='float32')
        Met.create_dataset('MT2_min', data=h5fw['INPUTS']['Met']['MT2_min'],dtype='float32') 
        Met.create_dataset('HT', data=h5fw['INPUTS']['Met']['HT'],dtype='float32') 
        Met.create_dataset('nbjet', data=h5fw['INPUTS']['Met']['nbjet'],dtype='int64') 
        Met.create_dataset('nljet', data=h5fw['INPUTS']['Met']['nljet'],dtype='int64')
        Met.create_dataset('nlargejet', data=h5fw['INPUTS']['Met']['nlargejet'],dtype='int64') 
        Met.create_dataset('nVx', data=h5fw['INPUTS']['Met']['nVx'],dtype='int64')
        Met.create_dataset('lepflav1', data=h5fw['INPUTS']['Met']['lepflav1'],dtype='int64')
        Met.create_dataset('M1', data=h5fw['INPUTS']['Met']['M1'],dtype='float32')   
        Met.create_dataset('M2', data=h5fw['INPUTS']['Met']['M2'],dtype='float32')
        Met.create_dataset('p1', data=h5fw['INPUTS']['Met']['p1'],dtype='float32')   
        Met.create_dataset('p2', data=h5fw['INPUTS']['Met']['p2'],dtype='float32')   
        targets_group = out_file.create_group('TARGETS')
        ht = targets_group.create_group(f'ht')
        ht.create_dataset('b', data=h5fw['TARGETS']['ht']['b'],dtype='int64')
        ht.create_dataset('q1', data=h5fw['TARGETS']['ht']['q1'],dtype='int64')
        ht.create_dataset('q2', data=h5fw['TARGETS']['ht']['q2'],dtype='int64')
        lt = targets_group.create_group(f'lt')
        lt.create_dataset('b', data=h5fw['TARGETS']['lt']['b'],dtype='int64')
        lt.create_dataset('l', data=h5fw['TARGETS']['lt']['l'],dtype='int64')
        regressions_group = out_file.create_group('REGRESSIONS')
        regression = regressions_group.create_group(f'EVENT')
        weights_group = out_file.create_group('WEIGHTS')
        weights = weights_group.create_group(f'EVENT')
        weights.create_dataset('event_weights',data=event_weights,dtype='float32')
        truth_info_group = out_file.create_group('truth_info')
        for info in h5fw['truth_info'].keys():
            truth_info_group.create_dataset(info, data=h5fw['truth_info'][info])
with h5py.File("/raven/u/mvigl/Stop/run/pre/SPANet_multi_class/spanet_inputs_odd_weighted.h5", 'r') as h5fw:

    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
    M12 = np.array([
                    h5fw['INPUTS']['Met']['M1'][:][y==1],
                    h5fw['INPUTS']['Met']['M2'][:][y==1],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    p12 = np.array([
                    h5fw['INPUTS']['Met']['p1'][:][y==1],
                    h5fw['INPUTS']['Met']['p2'][:][y==1],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    event_weights = h5fw['WEIGHTS']['EVENT']['event_weights'][:]
    multi_class = h5fw['CLASSIFICATIONS']['EVENT']['class'][:]
    print(h5fw['INPUTS']['Met']['p1'][:][y==0])
    print(h5fw['INPUTS']['Met']['p2'][:][y==0])
if __name__ == "__main__":
    for i,m in enumerate(masses):
        print(m)
        print( p12[((np.sum(M12==masses_slice[i],axis=-1)==2)).reshape(-1)] )
        print('unweighted : ', len(event_weights[y==1][((np.sum(M12==masses_slice[i],axis=-1)==2)).reshape(-1)]))
        print('weighted : ', np.sum(event_weights[y==1][((np.sum(M12==masses_slice[i],axis=-1)==2)).reshape(-1)]) )
    print('total N events : ', len(y))
    print('unweighted N signal : ', np.sum(y==1))
    print('unweighted N bkg : ', np.sum(y==0))
    bkg = np.sum(event_weights[(y==0)])
    sig = np.sum(event_weights[(y==1)])
    print('bkg : ', bkg)
    print('sig : ', sig)
    print('bkg/sig : ', bkg/sig)
    for i,m in enumerate(params):
        print(m)
        print('unweighted : ', len(event_weights[y==1][((np.sum(p12==params_slice[i],axis=-1)==2)).reshape(-1)]))
        print('weighted : ',  np.sum(event_weights[y==1][((np.sum(p12==params_slice[i],axis=-1)==2)).reshape(-1)]) )
    for c in range(5):
        print(c)
        print('unweighted : ',len(event_weights[multi_class==c]))
        print('weighted : ', np.sum(event_weights[multi_class==c]) )
