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

from typing import List
import numba
from numba import njit

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

TArray = np.ndarray

TFloat32 = numba.types.float32
TInt64 = numba.types.int64

TPrediction = numba.typed.typedlist.ListType(TFloat32[::1])
TPredictions = numba.typed.typedlist.ListType(TFloat32[:, ::1])

TResult = TInt64[:, ::1]
TResults = TInt64[:, :, ::1]

NUMBA_DEBUG = False


if NUMBA_DEBUG:
    def njit(*args, **kwargs):
        def wrapper(function):
            return function
        return wrapper


@njit("void(float32[::1], int64, int64, float32)")
def mask_1(data, size, index, value):
    data[index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_2(flat_data, size, index, value):
    data = flat_data.reshape((size, size))
    data[index, :] = value
    data[:, index] = value


@njit("void(float32[::1], int64, int64, float32)")
def mask_3(flat_data, size, index, value):
    data = flat_data.reshape((size, size, size))
    data[index, :, :] = value
    data[:, index, :] = value
    data[:, :, index] = value


@njit("void(float32[::1], int64, int64, int64, float32)")
def mask_jet(data, num_partons, max_jets, index, value):
    if num_partons == 1:
        mask_1(data, max_jets, index, value)
    elif num_partons == 2:
        mask_2(data, max_jets, index, value)
    elif num_partons == 3:
        mask_3(data, max_jets, index, value)


@njit("int64[::1](int64, int64)")
def compute_strides(num_partons, max_jets):
    strides = np.zeros(num_partons, dtype=np.int64)
    strides[-1] = 1
    for i in range(num_partons - 2, -1, -1):
        strides[i] = strides[i + 1] * max_jets

    return strides


@njit(TInt64[::1](TInt64, TInt64[::1]))
def unravel_index(index, strides):
    num_partons = strides.shape[0]
    result = np.zeros(num_partons, dtype=np.int64)

    remainder = index
    for i in range(num_partons):
        result[i] = remainder // strides[i]
        remainder %= strides[i]
    return result


@njit(TInt64(TInt64[::1], TInt64[::1]))
def ravel_index(index, strides):
    return (index * strides).sum()


@njit(numba.types.Tuple((TInt64, TInt64, TFloat32))(TPrediction))
def maximal_prediction(predictions):
    best_jet = -1
    best_prediction = -1
    best_value = -np.float32(np.inf)

    for i in range(len(predictions)):
        max_jet = np.argmax(predictions[i])
        max_value = predictions[i][max_jet]

        if max_value > best_value:
            best_prediction = i
            best_value = max_value
            best_jet = max_jet

    return best_jet, best_prediction, best_value


@njit(TResult(TPrediction, TInt64[::1], TInt64))
def extract_prediction(predictions, num_partons, max_jets):
    float_negative_inf = -np.float32(np.inf)
    max_partons = num_partons.max()
    num_targets = len(predictions)

    # Create copies of predictions for safety and calculate the output shapes
    strides = []
    for i in range(num_targets):
        strides.append(compute_strides(num_partons[i], max_jets))

    # Fill up the prediction matrix
    # -2 : Not yet assigned
    # -1 : Masked value
    # else : The actual index value
    results = np.zeros((num_targets, max_partons), np.int64) - 2

    for _ in range(num_targets):
        best_jet, best_prediction, best_value = maximal_prediction(predictions)

        if not np.isfinite(best_value):
            return results

        best_jets = unravel_index(best_jet, strides[best_prediction])

        results[best_prediction, :] = -1
        for i in range(num_partons[best_prediction]):
            results[best_prediction, i] = best_jets[i]

        predictions[best_prediction][:] = float_negative_inf
        for i in range(num_targets):
            for jet in best_jets:
                mask_jet(predictions[i], num_partons[i], max_jets, jet, float_negative_inf)

    return results


@njit(TResults(TPredictions, TInt64[::1], TInt64, TInt64), parallel=True)
def _extract_predictions(predictions, num_partons, max_jets, batch_size):
    output = np.zeros((batch_size, len(predictions), num_partons.max()), np.int64)
    predictions = [p.copy() for p in predictions]

    for batch in numba.prange(batch_size):
        current_prediction = numba.typed.List([prediction[batch] for prediction in predictions])
        #print('current_prediction :',current_prediction)
        output[batch, :, :] = extract_prediction(current_prediction, num_partons, max_jets)

    return np.ascontiguousarray(output.transpose((1, 0, 2)))


def extract_predictions(predictions: List[TArray]):
    flat_predictions = numba.typed.List([p.reshape((p.shape[0], -1)) for p in predictions])
    num_partons = np.array([len(p.shape) - 1 for p in predictions])
    print(num_partons)
    max_jets = max(max(p.shape[1:]) for p in predictions)
    batch_size = max(p.shape[0] for p in predictions)
    print(max_jets)

    results = _extract_predictions(flat_predictions, num_partons, max_jets, batch_size)
    return [result[:, :partons] for result, partons in zip(results, num_partons)]


def get_observable(pt,phi,eta,mass,predictions,mask_predictions=None,detection_probabilities=None,thr=0.2,reco='top',obs='mass'):
    pt = pt[np.arange(len(predictions))[:, np.newaxis], predictions]
    phi = phi[np.arange(len(predictions))[:, np.newaxis], predictions]
    eta = eta[np.arange(len(predictions))[:, np.newaxis], predictions]
    mass = mass[np.arange(len(predictions))[:, np.newaxis], predictions]
    mask = mask_predictions[np.arange(len(predictions))[:, np.newaxis], predictions]
    #for v in [pt,phi,eta,mass]:
        #v[mask==False] = 0.
        #v[detection_probabilities<thr,2] = 0
    b= vector.array(
        {
            "pt": pt[:,0],
            "phi": phi[:,0],
            "eta": eta[:,0],
            "M": mass[:,0],
        }
    )
    j1 = vector.array(
        {
            "pt": pt[:,1],
            "phi": phi[:,1],
            "eta": eta[:,1],
            "M": mass[:,1],
        }
    )
    j2 = vector.array(
        {
            "pt": pt[:,2],
            "phi": phi[:,2],
            "eta": eta[:,2],
            "M": mass[:,2],
        }
    )
    if reco == 'top': obj = b+(j1+j2)
    elif reco == 'W': obj = (j1+j2)
    elif reco == 'W_pair': obj = (j1)
    elif reco == 'top_pair': obj = b+(j1)
    else: 
        print('choose reco: top, W')
        return 0
    if obs=='mass': observable = obj.mass
    elif obs=='pt': observable = obj.pt
    else: 
        print('choose observable: mass')
        return 0
    return observable

def get_observable_leptop(pt,phi,eta,mass,predictions,mask_predictions=None,detection_probabilities=None,thr=0.2,reco='top',obs='mass'):
    pt = pt[np.arange(len(predictions))[:, np.newaxis], predictions]
    phi = phi[np.arange(len(predictions))[:, np.newaxis], predictions]
    eta = eta[np.arange(len(predictions))[:, np.newaxis], predictions]
    mass = mass[np.arange(len(predictions))[:, np.newaxis], predictions]
    #mask = mask_predictions[np.arange(len(predictions))[:, np.newaxis], predictions]
    #for v in [pt,phi,eta,mass]:
        #v[mask==False] = 0.
        #v[detection_probabilities<thr,2] = 0
    b = vector.array(
        {
            "pt": pt[:,0],
            "phi": phi[:,0],
            "eta": eta[:,0],
            "M": mass[:,0],
        }
    )

    l = vector.array(
        {
            "pt": pt[:,1],
            "phi": phi[:,1],
            "eta": eta[:,1],
            "M": mass[:,1],
        }
    )
    obj = (b+l)
    if obs=='mass': observable = obj.mass
    elif obs=='pt': observable = obj.pt
    return observable

session = onnxruntime.InferenceSession(
    "/raven/u/mvigl/TopReco/SPANet/spanet_log_norm.onnx", 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs:", [input.name for input in session.get_inputs()])
print("Inputs shape:", [input.shape for input in session.get_inputs()])
print("Outputs:", [output.name for output in session.get_outputs()])
print("Outputs shape:", [output.shape for output in session.get_outputs()])


import onnx
onnx_model = onnx.load("/raven/u/mvigl/TopReco/SPANet/spanet_log_norm.onnx")
onnx.checker.check_model(onnx_model)

with h5py.File("/raven//u/mvigl/Stop/run/pre/SPANet_all_8_cat_final/spanet_inputs_test.h5",'r') as h5fw :   
    #y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:40689]
    pt = h5fw['INPUTS']['Momenta']['pt'][:40689]
    eta = h5fw['INPUTS']['Momenta']['eta'][:40689]
    phi = h5fw['INPUTS']['Momenta']['phi'][:40689]
    mass = h5fw['INPUTS']['Momenta']['mass'][:40689]
    masks = h5fw['INPUTS']['Momenta']['MASK'][:40689]
    targets = np.column_stack((h5fw['TARGETS']['ht']['b'][:40689],h5fw['TARGETS']['ht']['q1'][:40689],h5fw['TARGETS']['ht']['q2'][:40689]))
    targets_lt = h5fw['TARGETS']['lt']['b'][:40689]
    targets_lt = targets_lt.reshape((len(targets_lt),-1))
    targets_lt = np.concatenate((targets_lt,np.ones(len(targets_lt)).reshape(len(targets_lt),-1)*7),axis=-1).astype(int)
    match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][:40689]
    nbs = h5fw['truth_info']['nbjet'][:40689]
    is_matched = h5fw['truth_info']['is_matched'][:40689]
     
    Momenta_data = np.array([h5fw['INPUTS']['Momenta']['mass'][:40689],
                    h5fw['INPUTS']['Momenta']['pt'][:40689],
                    h5fw['INPUTS']['Momenta']['eta'][:40689],
                    h5fw['INPUTS']['Momenta']['phi'][:40689],
                    h5fw['INPUTS']['Momenta']['btag'][:40689],
                    h5fw['INPUTS']['Momenta']['qtag'][:40689],
                    h5fw['INPUTS']['Momenta']['etag'][:40689]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(h5fw['INPUTS']['Momenta']['MASK'][:40689]).astype(bool)

    Met_data = np.array([h5fw['INPUTS']['Met']['MET'][:40689],
                    h5fw['INPUTS']['Met']['METsig'][:40689],
                    h5fw['INPUTS']['Met']['METphi'][:40689],
                    h5fw['INPUTS']['Met']['MET_Soft'][:40689],
                    h5fw['INPUTS']['Met']['MET_Jet'][:40689],
                    h5fw['INPUTS']['Met']['MET_Ele'][:40689],
                    h5fw['INPUTS']['Met']['MET_Muon'][:40689],
                    h5fw['INPUTS']['Met']['mT_METl'][:40689],
                    h5fw['INPUTS']['Met']['dR_bb'][:40689],
                    h5fw['INPUTS']['Met']['dphi_METl'][:40689],
                    h5fw['INPUTS']['Met']['MT2_bb'][:40689],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:40689],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:40689],
                    h5fw['INPUTS']['Met']['MT2_min'][:40689],
                    h5fw['INPUTS']['Met']['HT'][:40689],
                    h5fw['INPUTS']['Met']['nbjet'][:40689],
                    h5fw['INPUTS']['Met']['nljet'][:40689],
                    h5fw['INPUTS']['Met']['nVx'][:40689]]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

print('Momenta_data : ', Momenta_data.shape)  
print('Momenta_mask : ', Momenta_mask.shape)  
print('Met_data : ', Met_data.shape)    
print('Met_mask : ', Met_mask.shape)   

inputs = {}
inputs['Momenta_data'] = Momenta_data
inputs['Momenta_mask'] = Momenta_mask
inputs['Met_data'] = Met_data
inputs['Met_mask'] = Met_mask

def baseline_target_vars(pt,targets):
    pt = pt[np.arange(len(targets))[:, np.newaxis], targets]
    return pt

with h5py.File("/raven//u/mvigl/Stop/run/pre/H5_samples_test/multiplets_test.h5",'r') as h5fw :   
    counts = np.array(h5fw['variables'][:40689,variables.index('counts')])
    multiplets = h5fw['multiplets'][:np.sum(counts).astype(int)]
    vars = h5fw['variables'][:np.sum(counts).astype(int)]
    labels = h5fw['labels'][:np.sum(counts).astype(int)]

ort_sess_baseline = ort.InferenceSession("/raven/u/mvigl/TopReco/SPANet/baseline.onnx")
outputs_baseline = ort_sess_baseline.run(None, {'l_x_': multiplets})   

multiplets_evt = (ak.unflatten(multiplets, counts.astype(int)))
labels_evt = ((ak.unflatten(ak.Array(labels), counts.astype(int)))[:,0]==1)
max_baseline = np.array(ak.max(ak.unflatten(ak.Array(outputs_baseline[0]), counts.astype(int)),axis=1)).reshape(-1)
max_baseline_idx = (ak.argmax(ak.unflatten(ak.Array(outputs_baseline[0]), counts.astype(int)),axis=1))
triplets_baseline = multiplets_evt[max_baseline_idx][:,:,2]>0
pair_baseline = multiplets_evt[max_baseline_idx][:,:,2]==0
target_pt_baseline = baseline_target_vars(pt,targets)

def get_best(outputs):
    #priority to had top
    max_idxs_linear_had_top = np.argmax(outputs[1].reshape(len(outputs[1]), -1), axis=1)
    max_idxs_multi_had_top = np.array([np.unravel_index(idx, (10, 10, 10)) for idx in max_idxs_linear_had_top])

    #priority to lep top
    max_idxs_linear_lep_top = np.argmax(outputs[0].reshape(len(outputs[0]), -1), axis=1)
    max_idxs_multi_lep_top = np.array([np.unravel_index(idx, (10)) for idx in max_idxs_linear_lep_top])

    masked_htop_output = np.copy(outputs[1])
    masked_ltop_output = np.copy(outputs[0])
    masked_htop_output_min = np.copy(outputs[1])
    masked_ltop_output_min = np.copy(outputs[0])
    htop_fisrt = outputs[3][:]>=outputs[2][:]
    ltop_fisrt = outputs[3][:]<outputs[2][:]

    for i,idx in enumerate(max_idxs_multi_lep_top):
        if ltop_fisrt[i]==True:
            masked_htop_output[i, idx, :, :] = 0
            masked_htop_output[i, :, idx, :] = 0
            masked_htop_output[i, :, :, idx] = 0
        masked_htop_output_min[i, idx, :, :] = 0
        masked_htop_output_min[i, :, idx, :] = 0
        masked_htop_output_min[i, :, :, idx] = 0    

    for j in range(3):
        for i,idx in enumerate(max_idxs_multi_had_top[:,j]):
            if htop_fisrt[i]==True:
                masked_ltop_output[i,idx] = 0
            masked_ltop_output_min[i,idx] = 0          
    
    masked_htop_output = np.argmax(masked_htop_output.reshape(len(outputs[1]), -1), axis=1)
    had_top = np.array([np.unravel_index(idx, (10, 10, 10)) for idx in masked_htop_output])

    masked_ltop_output = np.argmax(masked_ltop_output.reshape(len(outputs[0]), -1), axis=1)
    lep_top = np.array([np.unravel_index(idx, (10)) for idx in masked_ltop_output])

    masked_htop_output_min = np.argmax(masked_htop_output_min.reshape(len(outputs[1]), -1), axis=1)
    had_top_min = np.array([np.unravel_index(idx, (10, 10, 10)) for idx in masked_htop_output_min])

    masked_ltop_output_min = np.argmax(masked_ltop_output_min.reshape(len(outputs[0]), -1), axis=1)
    lep_top_min = np.array([np.unravel_index(idx, (10)) for idx in masked_ltop_output_min])

    return had_top, lep_top, max_idxs_multi_had_top, max_idxs_multi_lep_top, had_top_min, lep_top_min

ort_sess = ort.InferenceSession("/raven/u/mvigl/TopReco/SPANet/spanet_v2_log_norm.onnx")

outputs = ort_sess.run(None, {'Momenta_data': Momenta_data,
                              'Momenta_mask': Momenta_mask,
                              'Met_data': Met_data,
                              'Met_mask': Met_mask})

out = extract_predictions(outputs[:2])

matching= {
    6: 'hadronic top w/ matched b-jet and 2 l-jets',
    5: 'hadronic top w/ matched b-jet and 1 l-jets',
    4: 'hadronic top w/ matched b-jet',
    3: 'hadronic top w/ no matched b-jet and 2 l-jets',
    2: 'hadronic top w/ no matched b-jet and 1 l-jet',
    1: 'hadronic top w/ no matched jets',
    0: 'no had top',
}

def plot_single_categories(had_top_mass,had_top_mass_min,max_idxs_multi_had_top_mass,top,target_top,
                            w_mass,w_mass_min,max_idxs_multi_w_mass,w,target_w,
                            lep_top_mass,lep_top_mass_min,max_idxs_multi_lep_top_mass,ltop,target_ltop,
                            baseline_top_mass,baseline_W_mass,targets_lt,
                            match=match_label,out=out,y=y,sample='sig',obj='top',obs='mass',algo='SPANet',thr=0.,category=5,
                           colors=[  '#1f77b4',
                                     '#ff7f0e',
                                     '#2ca02c',
                                     '#d62728',
                                     '#9467bd',
                                     '#8c564b',
                                     '#e377c2',
                                     '#7f7f7f',
                                     '#bcbd22',
                                     '#17becf']):
    
    if obj=='top': b=np.linspace(50,400,60)
    elif obj=='leptop': b=np.linspace(0,400,60)
    elif obj=='W': b=np.linspace(0,200,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plt.title(f'{algo} {matching[category]} {sample}')
    if obj == 'leptop': plt.title(f'{algo} leptonic top {sample}')
    
    label = np.ones_like(top)
    if sample == 'sig': label = (y==1)
    elif sample == 'bkg': label = (y==0)

    if obj == 'top':  
        if category > 4: ax.hist(target_top,weights=1*(match==category)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(had_top_mass,weights=1*(match==category)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(top,weights=1*(match==category)*(label),histtype='step',label='Reco default (based on assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(max_idxs_multi_had_top_mass,weights=1*(match==category)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(had_top_mass_min,weights=1*(match==category)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
        ax.hist(baseline_top_mass,weights=1*(match==category)*(label),histtype='step',label='Reco baseline',density=False,bins=b,color=colors[5])
    elif obj == 'W':  
        if category > 4: ax.hist(target_w,weights=1*(match==category)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(w_mass,weights=1*(match==category)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(w,weights=1*(match==category)*(label),histtype='step',label='Reco default (based on assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(max_idxs_multi_w_mass,weights=1*(match==category)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(w_mass_min,weights=1*(match==category)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
        ax.hist(baseline_W_mass,weights=1*(match==category)*(label),histtype='step',label='Reco baseline',density=False,bins=b,color=colors[5])
    elif obj == 'leptop':  
        ax.hist(target_ltop,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(lep_top_mass,weights=1*(targets_lt[:,0]!=-1)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(ltop,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Reco default (based on assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(lep_top_mass_min,weights=1*(targets_lt[:,0]!=-1)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(max_idxs_multi_lep_top_mass,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
    else: 
        return    
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'had top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obj=='leptop': ax.set_xlabel(f'lep top cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': ax.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': ax.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': ax.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    if obs in ['detection_probability','prediction_probability','prediction_probability_lt','detection_probability_lt']: 
        ax.semilogy()
    #ax.semilogy()    

    out_dir = f'Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Single_Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Single_Categories/{obj}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Single_Categories/{obj}/{obs}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Single_Categories/{obj}/{obs}/{sample}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    if obj == 'leptop': fig.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}.png')
    else: fig.savefig(f'{out_dir}/{sample}_cat_{category}_{obj}_{obs}_{algo}.png')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def get_auc(targets,predictions,title):
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets).reshape(-1),predictions.reshape(-1))
    Auc_sig = auc(fpr_sig,tpr_sig)
    plt.plot(fpr_sig,tpr_sig,label=f'auc : {Auc_sig:.2f}')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{title}.pdf')

def get_auc_vs(targets_spanet,predictions_spanet,targets_base,predictions_base,title):
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_spanet),predictions_spanet)
    Auc_sig_spanet = auc(fpr_sig,tpr_sig)
    plt.plot(fpr_sig,tpr_sig,label=f'spanet auc : {Auc_sig_spanet:.2f}')
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_base),predictions_base)
    Auc_sig_base = auc(fpr_sig,tpr_sig)
    plt.plot(fpr_sig,tpr_sig,label=f'baseline auc : {Auc_sig_base:.2f}')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{title}.pdf')

if __name__ == "__main__":

    get_auc(labels,outputs_baseline[0],'baseline_auc')
    get_auc_vs(np.array(labels_evt).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2])),labels_evt,max_baseline,'tagging_5_6_spanet_vs_baseline')
    get_auc_vs(np.array(labels_evt[y==1]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==1],np.array(labels_evt[y==1]).astype(int),max_baseline[y==1],'tagging_5_6_spanet_vs_baseline_sig')
    get_auc_vs(np.array(labels_evt[y==0]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==0],np.array(labels_evt[y==0]).astype(int),max_baseline[y==0],'tagging_5_6_spanet_vs_baseline_bkg')

    print('baseline accuracy on pairs : ', np.sum((np.sum(target_pt_baseline[(match_label==5)][:,:2]==(np.array(multiplets_evt[max_baseline_idx][:,:,:2]).reshape(-1,2)[(match_label==5)]),axis=-1)==2)*pair_baseline[(match_label==5)])/np.sum(match_label==5))
    print('baseline accuracy on triplets : ', np.sum(
    ((np.sum(target_pt_baseline[(match_label==6)][:,:3]==(np.array(multiplets_evt[max_baseline_idx][:,:,:3]).reshape(-1,3)[(match_label==6)]),axis=-1)==3)
    +(np.sum(target_pt_baseline[(match_label==6)][:,[0,2,1]]==(np.array(multiplets_evt[max_baseline_idx][:,:,:3]).reshape(-1,3)[(match_label==6)]),axis=-1)==3))
    )/np.sum(match_label==6))

    had_top, lep_top, max_idxs_multi_had_top, max_idxs_multi_lep_top, had_top_min, lep_top_min = get_best(outputs)
    lep_top = np.concatenate((lep_top,np.ones(len(lep_top)).reshape(len(lep_top),-1)*7),axis=-1).astype(int)
    max_idxs_multi_lep_top = np.concatenate((max_idxs_multi_lep_top,np.ones(len(max_idxs_multi_lep_top)).reshape(len(max_idxs_multi_lep_top),-1)*7),axis=-1).astype(int)
    lep_top_min = np.concatenate((lep_top_min,np.ones(len(lep_top_min)).reshape(len(lep_top_min),-1)*7),axis=-1).astype(int)
    
    print('spanet accuracy on triplets : ', np.sum((np.sum(had_top[(match_label==6)][:,:3]==(targets[:,:3])[(match_label==6)],axis=-1)==3))/np.sum(match_label==6) )
    print('spanet accuracy on pairs : ', np.sum((np.sum(had_top[(match_label==5)][:,[0,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
       +(np.sum(had_top[(match_label==5)][:,[1,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
       +(np.sum(had_top[(match_label==5)][:,[0,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
       +(np.sum(had_top[(match_label==5)][:,[2,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
       +(np.sum(had_top[(match_label==5)][:,[1,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
       +(np.sum(had_top[(match_label==5)][:,[2,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2))/np.sum(match_label==5) )

    if False:

        had_top_mass = get_observable(pt,phi,eta,mass,had_top,masks,thr=0.,reco='top',obs='mass')
        had_top_mass_min = get_observable(pt,phi,eta,mass,had_top_min,masks,thr=0.,reco='top',obs='mass')
        max_idxs_multi_had_top_mass = get_observable(pt,phi,eta,mass,max_idxs_multi_had_top,masks,thr=0.,reco='top',obs='mass')
        top = get_observable(pt,phi,eta,mass,out[1],masks,thr=0.,reco='top',obs='mass')
        target_top = get_observable(pt,phi,eta,mass,targets,masks,thr=0.,reco='top',obs='mass')

        w_mass = get_observable(pt,phi,eta,mass,had_top,masks,thr=0.,reco='W',obs='mass')
        w_mass_min = get_observable(pt,phi,eta,mass,had_top_min,masks,thr=0.,reco='W',obs='mass')
        max_idxs_multi_w_mass = get_observable(pt,phi,eta,mass,max_idxs_multi_had_top,masks,thr=0.,reco='W',obs='mass')
        w = get_observable(pt,phi,eta,mass,out[1],masks,thr=0.,reco='W',obs='mass')
        target_w = get_observable(pt,phi,eta,mass,targets,masks,thr=0.,reco='W',obs='mass')

        lep_top_mass = get_observable_leptop(pt,phi,eta,mass,lep_top,masks,thr=0.,reco='top',obs='mass')
        lep_top_mass_min = get_observable_leptop(pt,phi,eta,mass,lep_top_min,masks,thr=0.,reco='top',obs='mass')
        max_idxs_multi_lep_top_mass = get_observable_leptop(pt,phi,eta,mass,max_idxs_multi_lep_top,masks,thr=0.,reco='top',obs='mass')
        ltop = get_observable_leptop(pt,phi,eta,mass,np.concatenate((out[0],np.ones(len(out[0])).reshape(len(out[0]),-1)*7),axis=-1).astype(int),masks,thr=0.,reco='top',obs='mass')
        target_ltop = get_observable_leptop(pt,phi,eta,mass,targets_lt,masks,thr=0.,reco='top',obs='mass')

        for sample in ['all','sig','bkg']:
            for category in [6,3,0,1,2,4,5]:
                for obj in ['top','W','leptop']:
                    for obs in ['mass']:#,'pt']:
                        if (obj=='W' and obs=='pt'): continue
                        if (obj=='leptop' and category!=6): continue
                        plot_single_categories(had_top_mass,had_top_mass_min,max_idxs_multi_had_top_mass,top,target_top,
                                               w_mass,w_mass_min,max_idxs_multi_w_mass,w,target_w,
                                               lep_top_mass,lep_top_mass_min,max_idxs_multi_lep_top_mass,ltop,target_ltop,
                                               baseline_top_mass,baseline_W_mass,targets_lt,
                                                sample=sample,out=out,y=y,obj=obj,obs=obs,algo='SPANet',thr=0,category=category,colors=colors)  

   