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
    
    pt = h5fw['INPUTS']['Momenta']['pt'][:110000]
    eta = h5fw['INPUTS']['Momenta']['eta'][:110000]
    phi = h5fw['INPUTS']['Momenta']['phi'][:110000]
    mass = h5fw['INPUTS']['Momenta']['mass'][:110000]
    masks = h5fw['INPUTS']['Momenta']['MASK'][:110000]
    targets = np.column_stack((h5fw['TARGETS']['ht']['b'][:110000],h5fw['TARGETS']['ht']['q1'][:110000],h5fw['TARGETS']['ht']['q2'][:110000]))
    targets_lt = h5fw['TARGETS']['lt']['b'][:110000]
    targets_lt = targets_lt.reshape((len(targets_lt),-1))
    targets_lt = np.concatenate((targets_lt,np.ones(len(targets_lt)).reshape(len(targets_lt),-1)*7),axis=-1).astype(int)
    match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][:110000]
    nbs = h5fw['truth_info']['nbjet'][:110000]
    is_matched = h5fw['truth_info']['is_matched'][:110000]
     
    Momenta_data = np.array([h5fw['INPUTS']['Momenta']['mass'][:110000],
                    h5fw['INPUTS']['Momenta']['pt'][:110000],
                    h5fw['INPUTS']['Momenta']['eta'][:110000],
                    h5fw['INPUTS']['Momenta']['phi'][:110000],
                    h5fw['INPUTS']['Momenta']['btag'][:110000],
                    h5fw['INPUTS']['Momenta']['qtag'][:110000],
                    h5fw['INPUTS']['Momenta']['etag'][:110000]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(h5fw['INPUTS']['Momenta']['MASK'][:110000]).astype(bool)

    Met_data = np.array([h5fw['INPUTS']['Met']['MET'][:110000],
                    h5fw['INPUTS']['Met']['METsig'][:110000],
                    h5fw['INPUTS']['Met']['METphi'][:110000],
                    h5fw['INPUTS']['Met']['MET_Soft'][:110000],
                    h5fw['INPUTS']['Met']['MET_Jet'][:110000],
                    h5fw['INPUTS']['Met']['MET_Ele'][:110000],
                    h5fw['INPUTS']['Met']['MET_Muon'][:110000],
                    h5fw['INPUTS']['Met']['mT_METl'][:110000],
                    h5fw['INPUTS']['Met']['dR_bb'][:110000],
                    h5fw['INPUTS']['Met']['dphi_METl'][:110000],
                    h5fw['INPUTS']['Met']['MT2_bb'][:110000],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:110000],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:110000],
                    h5fw['INPUTS']['Met']['MT2_min'][:110000],
                    h5fw['INPUTS']['Met']['HT'][:110000],
                    h5fw['INPUTS']['Met']['nbjet'][:110000],
                    h5fw['INPUTS']['Met']['nljet'][:110000],
                    h5fw['INPUTS']['Met']['nVx'][:110000],
                    h5fw['INPUTS']['Met']['M1'][:110000],
                    h5fw['INPUTS']['Met']['M2'][:110000],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:110000]

with h5py.File("/raven//u/mvigl/Stop/run/pre/SPANet_all_8_cat_final/spanet_inputs_test.h5",'r') as h5fw :      
    Met_data_base = np.array([h5fw['INPUTS']['Met']['MET'][:110000],
                    h5fw['INPUTS']['Met']['METsig'][:110000],
                    h5fw['INPUTS']['Met']['METphi'][:110000],
                    h5fw['INPUTS']['Met']['MET_Soft'][:110000],
                    h5fw['INPUTS']['Met']['MET_Jet'][:110000],
                    h5fw['INPUTS']['Met']['MET_Ele'][:110000],
                    h5fw['INPUTS']['Met']['MET_Muon'][:110000],
                    h5fw['INPUTS']['Met']['mT_METl'][:110000],
                    h5fw['INPUTS']['Met']['dR_bb'][:110000],
                    h5fw['INPUTS']['Met']['dphi_METl'][:110000],
                    h5fw['INPUTS']['Met']['MT2_bb'][:110000],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:110000],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:110000],
                    h5fw['INPUTS']['Met']['MT2_min'][:110000],
                    h5fw['INPUTS']['Met']['HT'][:110000],
                    h5fw['INPUTS']['Met']['nbjet'][:110000],
                    h5fw['INPUTS']['Met']['nljet'][:110000],
                    h5fw['INPUTS']['Met']['nVx'][:110000],]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
print('Momenta_data : ', Momenta_data.shape)  
print('Momenta_mask : ', Momenta_mask.shape)  
print('Met_data : ', Met_data.shape)    
print('Met_mask : ', Met_mask.shape)    

inputs = {}
inputs['Met_data'] = {}
inputs['Momenta_data'] = np.copy(Momenta_data)
inputs['Momenta_mask'] = np.copy(Momenta_mask)
inputs['Met_data']['1000_100'] = np.copy(Met_data)
inputs['Met_data']['1000_200'] = np.copy(Met_data)
inputs['Met_data']['1000_300'] = np.copy(Met_data)
inputs['Met_data']['1000_400'] = np.copy(Met_data)
inputs['Met_data']['1000_500'] = np.copy(Met_data)
inputs['Met_data']['1000_600'] = np.copy(Met_data)
inputs['Met_data']['1000_700'] = np.copy(Met_data)
inputs['Met_data']['1000_800'] = np.copy(Met_data)
inputs['Met_mask'] = np.copy(Met_mask)
(inputs['Met_data']['1000_100'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  100.]
(inputs['Met_data']['1000_200'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  200.]
(inputs['Met_data']['1000_300'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  300.]
(inputs['Met_data']['1000_400'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  400.]
(inputs['Met_data']['1000_500'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  500.]
(inputs['Met_data']['1000_600'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  600.]
(inputs['Met_data']['1000_700'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  700.]
(inputs['Met_data']['1000_800'][:,:,-2:])[np.sum(Met_data[:,:,-2:]==[-1,-1],axis=-1)==2]=[1000.,  800.]

inputs_base = {}
inputs_base['Momenta_data'] = np.copy(Momenta_data)
inputs_base['Momenta_mask'] = np.copy(Momenta_mask)
inputs_base['Met_data']= np.copy(Met_data_base)
inputs_base['Met_mask'] = np.copy(Met_mask)


ort_sess_base = ort.InferenceSession("/Users/matthiasvigl/Documents/Physics/Stop/TopReco/SPANet_stop1L/spanet_v2_log_norm.onnx")
outputs_base = ort_sess_base.run(None, {'Momenta_data': inputs_base['Momenta_data'],
                              'Momenta_mask': inputs_base['Momenta_mask'],
                              'Met_data': inputs_base['Met_data'],
                              'Met_mask': inputs_base['Met_mask']})


ort_sess = ort.InferenceSession("/Users/matthiasvigl/Documents/Physics/Stop/TopReco/SPANet_stop1L/spanet_param_log_norm.onnx")
outputs = {}
for masses in ['1000_100','1000_200','1000_300','1000_400','1000_500','1000_600','1000_700','1000_800']:
    outputs[masses] = ort_sess.run(None, {'Momenta_data': inputs['Momenta_data'],
                              'Momenta_mask': inputs['Momenta_mask'],
                              'Met_data': inputs['Met_data'][masses],
                              'Met_mask': inputs['Met_mask']})

if __name__ == "__main__":

    masses = ['1000_100','1000_200','1000_300','1000_400','1000_500','1000_600','1000_700','1000_800']
    masses_slice = [[1000,100],[1000,200],[1000,300],[1000,400],[1000,500],[1000,600],[1000,700],[1000,800]]
    auc_base = np.zeros(len(masses))
    auc_par = np.zeros(len(masses))
    for i,m in enumerate(masses):
        b = np.linspace(0,1,30)
        param_filter = (np.sum(inputs['Met_data'][m][:,:,-2:]==masses_slice[i],axis=-1)==2).reshape(-1)
        plt.hist(outputs[m][4][:,1],weights=1*(y==1)*param_filter,histtype='step',label=f'sig {m}',density=True,bins=b)
        plt.hist(outputs[m][4][:,1],weights=1*(y==0)*param_filter,histtype='step',label=f'bkg {m}',density=True,bins=b)
        plt.hist(outputs_base[4][:,1],weights=1*(y==1)*param_filter,histtype='step',label=f'sig',density=True,bins=b)
        plt.hist(outputs_base[4][:,1],weights=1*(y==0)*param_filter,histtype='step',label=f'bkg',density=True,bins=b)
        fpr_sig, tpr_sig, thresholds_sig = roc_curve((y[param_filter]),outputs[m][4][param_filter,1])
        Auc_sig = auc(fpr_sig,tpr_sig)
        fpr_sig, tpr_sig, thresholds_sig = roc_curve((y[param_filter]),outputs_base[4][param_filter,1])
        Auc_sig_nonPar = auc(fpr_sig,tpr_sig)
        plt.title(f'auc par : {Auc_sig:.3f}   ,   auc base : {Auc_sig_nonPar:.3f}')
        plt.legend()
        plt.semilogy()
        plt.savefig(f'{m}.pdf')
        auc_par[i]=Auc_sig
        auc_base[i]=Auc_sig_nonPar