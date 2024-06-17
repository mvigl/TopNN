import h5py
import vector
import os
import onnxruntime
import vector 
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import argparse
import awkward as ak
import onnxruntime
from sklearn.metrics import roc_curve,auc
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', help='data',default='data/root/list_sig_FS_testing.txt')
parser.add_argument('--evals', help='evals',default='data/root/list_sig_FS_testing.txt')
args = parser.parse_args()


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

inputs_baseline = [  'bjet_pT',
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


with h5py.File("/raven//u/mvigl/Stop/run/pre/SPANet_all_8_cat_final/spanet_inputs_test.h5",'r') as h5fw :
    
    pt = h5fw['INPUTS']['Momenta']['pt'][:]
    eta = h5fw['INPUTS']['Momenta']['eta'][:]
    phi = h5fw['INPUTS']['Momenta']['phi'][:]
    mass = h5fw['INPUTS']['Momenta']['mass'][:]
    masks = h5fw['INPUTS']['Momenta']['MASK'][:]
    targets = np.column_stack((h5fw['TARGETS']['ht']['b'][:],h5fw['TARGETS']['ht']['q1'][:],h5fw['TARGETS']['ht']['q2'][:]))
    targets_lt = h5fw['TARGETS']['lt']['b'][:]
    targets_lt = targets_lt.reshape((len(targets_lt),-1))
    targets_lt = np.concatenate((targets_lt,np.ones(len(targets_lt)).reshape(len(targets_lt),-1)*7),axis=-1).astype(int)
    match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][:]
    nbs = h5fw['truth_info']['nbjet'][:]
    is_matched = h5fw['truth_info']['is_matched'][:]
     
    Momenta_data = np.array([h5fw['INPUTS']['Momenta']['mass'][:],
                    h5fw['INPUTS']['Momenta']['pt'][:],
                    h5fw['INPUTS']['Momenta']['eta'][:],
                    h5fw['INPUTS']['Momenta']['phi'][:],
                    h5fw['INPUTS']['Momenta']['btag'][:],
                    h5fw['INPUTS']['Momenta']['qtag'][:],
                    h5fw['INPUTS']['Momenta']['etag'][:]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(h5fw['INPUTS']['Momenta']['MASK'][:]).astype(bool)

    Met_data = np.array([h5fw['INPUTS']['Met']['MET'][:],
                    h5fw['INPUTS']['Met']['METsig'][:],
                    h5fw['INPUTS']['Met']['METphi'][:],
                    h5fw['INPUTS']['Met']['MET_Soft'][:],
                    h5fw['INPUTS']['Met']['MET_Jet'][:],
                    h5fw['INPUTS']['Met']['MET_Ele'][:],
                    h5fw['INPUTS']['Met']['MET_Muon'][:],
                    h5fw['INPUTS']['Met']['mT_METl'][:],
                    h5fw['INPUTS']['Met']['dR_bb'][:],
                    h5fw['INPUTS']['Met']['dphi_METl'][:],
                    h5fw['INPUTS']['Met']['MT2_bb'][:],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:],
                    h5fw['INPUTS']['Met']['MT2_min'][:],
                    h5fw['INPUTS']['Met']['HT'][:],
                    h5fw['INPUTS']['Met']['nbjet'][:],
                    h5fw['INPUTS']['Met']['nljet'][:],
                    h5fw['INPUTS']['Met']['nVx'][:],
                    h5fw['INPUTS']['Met']['M1'][:],
                    h5fw['INPUTS']['Met']['M2'][:],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]


def run_in_batches(model_path, Momenta_data,Momenta_mask,Met_data,Met_mask, batch_size, masses,masses_slice):
    ort_sess = ort.InferenceSession(model_path)
    
    outputs = {}
    num_batches = len(Met_data) // batch_size
    if len(Met_data) % batch_size != 0:
        num_batches += 1
    print('num batches : ',num_batches)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(Met_data))
        print('batch : ',i,'/',num_batches)
        for j,mass in enumerate(masses):
            batch_inputs = {
                'Momenta_data': Momenta_data[start_idx:end_idx],
                'Momenta_mask': Momenta_mask[start_idx:end_idx],
                'Met_data': Met_data[start_idx:end_idx],
                'Met_mask': Met_mask[start_idx:end_idx]
            }
            (batch_inputs['Met_data'][:,:,-2:])[np.sum(Met_data[start_idx:end_idx,:,-2:]==[-1,-1],axis=-1)==2]=masses_slice[j]
            outputs[mass] = ort_sess.run(None, {'Momenta_data': batch_inputs['Momenta_data'],
                              'Momenta_mask': batch_inputs['Momenta_mask'],
                              'Met_data': batch_inputs['Met_data'],
                              'Met_mask': batch_inputs['Met_mask']})
        
        with open(f'evals_param_batch_{i}.pkl', 'wb') as pickle_file:
            pickle.dump(outputs, pickle_file)    
    return outputs


batch_size = 100000  # Adjust batch size based on your memory constraints

masses = ['500_1','500_100','500_200','500_300',
          '600_1','600_100','600_200','600_300','600_400',
          '700_1','700_100','700_200','700_300','700_400','700_500',
          '800_1','800_100','800_200','800_300','800_400','800_500','800_600',
          '900_1','900_100','900_200','900_300','900_400','900_500','900_600','900_700',
          '1000_1','1000_100','1000_200','1000_300','1000_400','1000_500','1000_600','1000_700','1000_800',
          '1100_1','1100_100','1100_200','1100_300','1100_400','1100_500','1100_600','1100_700','1100_800',
          '1200_1','1200_100','1200_200','1200_300','1200_400','1200_500','1200_600','1200_700','1200_800',
          '1300_1','1300_100','1300_200','1300_300','1300_400','1300_500','1300_600','1300_700','1300_800',
          '1400_1','1400_100','1400_200','1400_300','1400_400','1400_500','1400_600','1400_700','1400_800',
          '1500_1','1500_100','1500_200','1500_300','1500_400','1500_500','1500_600','1500_700','1500_800',
          '1600_1','1600_100','1600_200','1600_300','1600_400','1600_500','1600_600','1600_700','1600_800']
masses_slice = [[500,1],[500,100],[500,200],[500,300],
                [600,1],[600,100],[600,200],[600,300],[600,400],
                [700,1],[700,100],[700,200],[700,300],[700,400],[700,500],
                [800,1],[800,100],[800,200],[800,300],[800,400],[800,500],[800,600],
                [900,1],[900,100],[900,200],[900,300],[900,400],[900,500],[900,600],[900,700],
                [1000,1],[1000,100],[1000,200],[1000,300],[1000,400],[1000,500],[1000,600],[1000,700],[1000,800],
                [1100,1],[1100,100],[1100,200],[1100,300],[1100,400],[1100,500],[1100,600],[1100,700],[1100,800],
                [1200,1],[1200,100],[1200,200],[1200,300],[1200,400],[1200,500],[1200,600],[1200,700],[1200,800],
                [1300,1],[1300,100],[1300,200],[1300,300],[1300,400],[1300,500],[1300,600],[1300,700],[1300,800],
                [1400,1],[1400,100],[1400,200],[1400,300],[1400,400],[1400,500],[1400,600],[1400,700],[1400,800],
                [1500,1],[1500,100],[1500,200],[1500,300],[1500,400],[1500,500],[1500,600],[1500,700],[1500,800],
                [1600,1],[1600,100],[1600,200],[1600,300],[1600,400],[1600,500],[1600,600],[1600,700],[1600,800]]

if __name__ == "__main__":

   outputs = run_in_batches("/raven/u/mvigl/TopReco/SPANet/spanet_param_log_norm.onnx", Momenta_data,Momenta_mask,Met_data,Met_mask,batch_size,masses,masses_slice)
