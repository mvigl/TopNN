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
parser.add_argument('--evals', help='evals',default='0_1')
args = parser.parse_args()

from typing import List
import numba
from numba import njit

import mplhep as hep
hep.style.use([hep.style.ATLAS])

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


def get_observable(pt,phi,eta,mass,predictions,mask_predictions=None,cut_prob=None,thr=0.,reco='top',obs='mass'):
    pt = pt[np.arange(len(predictions))[:, np.newaxis], predictions]
    phi = phi[np.arange(len(predictions))[:, np.newaxis], predictions]
    eta = eta[np.arange(len(predictions))[:, np.newaxis], predictions]
    mass = mass[np.arange(len(predictions))[:, np.newaxis], predictions]
    mask = mask_predictions[np.arange(len(predictions))[:, np.newaxis], predictions]
    if cut_prob is not None:
        for v in [pt,phi,eta,mass]:
            #v[mask==False] = 0.
            v[cut_prob>thr,2] = 0
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
    elif obs=='eta': observable = obj.eta
    elif obs=='phi': observable = obj.phi
    else: 
        print('choose observable: mass')
        return 0
    return obj#observable

def get_observable_leptop(pt,phi,eta,mass,predictions,mask_predictions=None,detection_probabilities=None,thr=0.,reco='top',obs='mass'):
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
    return obj#observable

session = onnxruntime.InferenceSession(
    "/raven/u/mvigl/TopReco/SPANet/spanet_multiclass_even.onnx", 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs:", [input.name for input in session.get_inputs()])
print("Inputs shape:", [input.shape for input in session.get_inputs()])
print("Outputs:", [output.name for output in session.get_outputs()])
print("Outputs shape:", [output.shape for output in session.get_outputs()])


import onnx
onnx_model = onnx.load("/raven/u/mvigl/TopReco/SPANet/spanet_multiclass_even.onnx")
onnx.checker.check_model(onnx_model)

with h5py.File("/raven//u/mvigl/Stop/run/pre/SPANet_multi_class/spanet_inputs_odd_weighted.h5",'r') as h5fw :   
    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
    subset = (y==1).reshape(-1).astype(bool)
    y=y[subset]
    pt = h5fw['INPUTS']['Momenta']['pt'][subset]
    eta = h5fw['INPUTS']['Momenta']['eta'][subset]
    phi = h5fw['INPUTS']['Momenta']['phi'][subset]
    mass = h5fw['INPUTS']['Momenta']['mass'][subset]
    print(h5fw['INPUTS']['Momenta']['mass'].shape)
    print(h5fw['INPUTS']['Momenta']['MASK'].shape)
    print(subset.shape)
    print(subset)
    print(h5fw['INPUTS']['Momenta']['MASK'][:][subset].shape)
    masks = h5fw['INPUTS']['Momenta']['MASK'][:][subset]
    targets = np.column_stack((h5fw['TARGETS']['ht']['b'][subset],h5fw['TARGETS']['ht']['q1'][subset],h5fw['TARGETS']['ht']['q2'][subset]))
    targets_lt = h5fw['TARGETS']['lt']['b'][subset]
    targets_lt = targets_lt.reshape((len(targets_lt),-1))
    targets_lt = np.concatenate((targets_lt,np.ones(len(targets_lt)).reshape(len(targets_lt),-1)*7),axis=-1).astype(int)
    match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][subset]
    nbs = h5fw['truth_info']['nbjet'][subset]
    is_matched = h5fw['truth_info']['is_matched'][subset]
    event_weights = h5fw['truth_info']['training_weights'][subset]
    train_weights = h5fw['WEIGHTS']['EVENT']['event_weights'][subset]
     
    Momenta_data = np.array([h5fw['INPUTS']['Momenta']['mass'][subset],
                    h5fw['INPUTS']['Momenta']['pt'][subset],
                    h5fw['INPUTS']['Momenta']['eta'][subset],
                    h5fw['INPUTS']['Momenta']['phi'][subset],
                    h5fw['INPUTS']['Momenta']['btag'][subset],
                    h5fw['INPUTS']['Momenta']['qtag'][subset],
                    h5fw['INPUTS']['Momenta']['etag'][subset],
                    h5fw['INPUTS']['Momenta']['bscore'][subset],
                    h5fw['INPUTS']['Momenta']['Larget'][subset],
                    h5fw['INPUTS']['Momenta']['LargeZ'][subset],
                    h5fw['INPUTS']['Momenta']['LargeW'][subset]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(h5fw['INPUTS']['Momenta']['MASK'][:][subset]).astype(bool)

    Met_data = np.array([h5fw['INPUTS']['Met']['MET'][subset],
                    h5fw['INPUTS']['Met']['METsig'][subset],
                    h5fw['INPUTS']['Met']['METphi'][subset],
                    #h5fw['INPUTS']['Met']['MET_Soft'][subset],
                    #h5fw['INPUTS']['Met']['MET_Jet'][subset],
                    #h5fw['INPUTS']['Met']['MET_Ele'][subset],
                    #h5fw['INPUTS']['Met']['MET_Muon'][subset],
                    h5fw['INPUTS']['Met']['mT_METl'][subset],
                    h5fw['INPUTS']['Met']['dR_bb'][subset],
                    h5fw['INPUTS']['Met']['dphi_METl'][subset],
                    h5fw['INPUTS']['Met']['MT2_bb'][subset],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][subset],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][subset],
                    h5fw['INPUTS']['Met']['MT2_min'][subset],
                    h5fw['INPUTS']['Met']['HT'][subset],
                    h5fw['INPUTS']['Met']['nbjet'][subset],
                    h5fw['INPUTS']['Met']['nljet'][subset],
                    h5fw['INPUTS']['Met']['nVx'][subset],
                    h5fw['INPUTS']['Met']['lepflav1'][subset]]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

print('Momenta_data : ', Momenta_data.shape)  
print('Momenta_mask : ', Momenta_mask.shape)  
print('Met_data : ', Met_data.shape)    
print('Met_mask : ', Met_mask.shape)   

def baseline_target_vars(pt,targets):
    pt = pt[np.arange(len(targets))[:, np.newaxis], targets]
    return pt

#with h5py.File("/raven/u/mvigl/Stop/run/pre/H5_samples_odd/multiplets_odd.h5",'r') as h5fw :   
#    counts = np.array(h5fw['variables'][:,variables.index('counts')])[subset]
#    print(counts)
#    multiplets = h5fw['multiplets'][:int(np.sum(counts))]
#    labels = h5fw['labels'][:int(np.sum(counts))]
#
#ort_sess_baseline = ort.InferenceSession("/raven/u/mvigl/TopReco/SPANet/baseline.onnx")
#outputs_baseline = ort_sess_baseline.run(None, {'l_x_': multiplets})   
#with open('/raven/u/mvigl/Stop/run/spanet_vs_baseline/evals_baseline_test.pkl', 'wb') as pickle_file:
#    pickle.dump(outputs_baseline, pickle_file)   
#
#with open('/raven/u/mvigl/Stop/run/spanet_vs_baseline/evals_baseline_test.pkl', 'rb') as pickle_file:
#    outputs_baseline = pickle.load(pickle_file)        

#multiplets_evt = (ak.unflatten(multiplets, counts.astype(int)))
labels_evt = is_matched#((ak.unflatten(ak.Array(labels), counts.astype(int)))[:,0]==1)
#max_baseline = np.array(ak.max(ak.unflatten(ak.Array(outputs_baseline[0]), counts.astype(int)),axis=1)).reshape(-1)
#max_baseline_idx = (ak.argmax(ak.unflatten(ak.Array(outputs_baseline[0]), counts.astype(int)),axis=1))
#triplets_baseline = multiplets_evt[max_baseline_idx][:,:,2]>0
#pair_baseline = multiplets_evt[max_baseline_idx][:,:,2]==0
#target_pt_baseline = baseline_target_vars(pt,targets)
#baseline_preds = multiplets_evt[max_baseline_idx]

def get_observable_baseline(baseline_preds,reco='top',obs='mass'):
    b= vector.array(
        {
            "pt": np.array(baseline_preds[:,:,inputs_baseline.index('bjet_pT')]).reshape(-1),
            "phi": np.array(baseline_preds[:,:,inputs_baseline.index('bjet_phi')]).reshape(-1),
            "eta": np.array(baseline_preds[:,:,inputs_baseline.index('bjet_eta')]).reshape(-1),
            "M": np.array(baseline_preds[:,:,inputs_baseline.index('bjet_M')]).reshape(-1),
        }
    )
    j1 = vector.array(
        {
            "pt": np.array(baseline_preds[:,:,inputs_baseline.index('jet1_pT')]).reshape(-1),
            "phi": np.array(baseline_preds[:,:,inputs_baseline.index('jet1_phi')]).reshape(-1),
            "eta": np.array(baseline_preds[:,:,inputs_baseline.index('jet1_eta')]).reshape(-1),
            "M": np.array(baseline_preds[:,:,inputs_baseline.index('jet1_M')]).reshape(-1),
        }
    )
    j2 = vector.array(
        {
            "pt": np.array(baseline_preds[:,:,inputs_baseline.index('jet2_pT')]).reshape(-1),
            "phi": np.array(baseline_preds[:,:,inputs_baseline.index('jet2_phi')]).reshape(-1),
            "eta": np.array(baseline_preds[:,:,inputs_baseline.index('jet2_eta')]).reshape(-1),
            "M": np.array(baseline_preds[:,:,inputs_baseline.index('jet2_M')]).reshape(-1),
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
    return obj#observable


def get_best(outputs):
    #priority to had top
    max_idxs_linear_had_top = np.argmax(outputs[1].reshape(len(outputs[1]), -1), axis=1)
    max_idxs_multi_had_top = np.array([np.unravel_index(idx, (13, 13, 13)) for idx in max_idxs_linear_had_top])

    #priority to lep top
    max_idxs_linear_lep_top = np.argmax(outputs[0].reshape(len(outputs[0]), -1), axis=1)
    max_idxs_multi_lep_top = np.array([np.unravel_index(idx, (13)) for idx in max_idxs_linear_lep_top])

    masked_htop_output = np.copy(outputs[1])
    masked_ltop_output = np.copy(outputs[0])
    masked_htop_output_min = np.copy(outputs[1])
    masked_ltop_output_min = np.copy(outputs[0])

    #priority based on detection prob
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
    had_top = np.array([np.unravel_index(idx, (13, 13, 13)) for idx in masked_htop_output])

    masked_ltop_output = np.argmax(masked_ltop_output.reshape(len(outputs[0]), -1), axis=1)
    lep_top = np.array([np.unravel_index(idx, (13)) for idx in masked_ltop_output])

    masked_htop_output_min = np.argmax(masked_htop_output_min.reshape(len(outputs[1]), -1), axis=1)
    had_top_min = np.array([np.unravel_index(idx, (13, 13, 13)) for idx in masked_htop_output_min])

    masked_ltop_output_min = np.argmax(masked_ltop_output_min.reshape(len(outputs[0]), -1), axis=1)
    lep_top_min = np.array([np.unravel_index(idx, (13)) for idx in masked_ltop_output_min])

    return had_top, lep_top, max_idxs_multi_had_top, max_idxs_multi_lep_top, had_top_min, lep_top_min



def run_in_batches(model_path, Momenta_data,Momenta_mask,Met_data,Met_mask, batch_size):
    ort_sess = ort.InferenceSession(model_path)
    
    outputs = []
    num_batches = len(Met_data) // batch_size
    if len(Met_data) % batch_size != 0:
        num_batches += 1
    print('num batches : ',num_batches)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(Met_data))
        print('batch : ',i,'/',num_batches)
        batch_inputs = {
            'Momenta_data': Momenta_data[start_idx:end_idx],
            'Momenta_mask': Momenta_mask[start_idx:end_idx],
            'Met_data': Met_data[start_idx:end_idx],
            'Met_mask': Met_mask[start_idx:end_idx]
        }
        
        batch_outputs = ort_sess.run(None, batch_inputs)
        if i == 0: outputs = batch_outputs
        else: 
            for j in range(len(outputs)):
                outputs[j]=np.concatenate((outputs[j],batch_outputs[j]),axis=0)
    
    return outputs

batch_size = 20000  # Adjust batch size based on your memory constraints

#outputs_old = run_in_batches("/raven/u/mvigl/TopReco/SPANet/spanet_weights_log_norm.onnx", Momenta_data,Momenta_mask,Met_data,Met_mask, batch_size)
#with open('evals_test_weights.pkl', 'wb') as pickle_file:
#    pickle.dump(outputs_old, pickle_file)
#
outputs = run_in_batches("/raven/u/mvigl/TopReco/SPANet/spanet_multiclass_even.onnx", Momenta_data,Momenta_mask,Met_data,Met_mask, batch_size)
with open('/raven/u/mvigl/Stop/run/spanet_vs_baseline/evals_test_multiclass.pkl', 'wb') as pickle_file:
    pickle.dump(outputs, pickle_file)
 
#with open('evals_test_weights.pkl', 'rb') as pickle_file:
#    outputs_old = pickle.load(pickle_file) 

with open('/raven/u/mvigl/Stop/run/spanet_vs_baseline/evals_test_multiclass.pkl', 'rb') as pickle_file:
    outputs = pickle.load(pickle_file) 

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
                            #baseline_top_mass,baseline_W_mass,
                            targets_lt,
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
                                     '#17becf'],
                                     mess=''):
    
    if obj=='top': b=np.linspace(50,400,60)
    elif obj=='leptop': b=np.linspace(0,400,60)
    elif obj=='W': b=np.linspace(0,150,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    elif obs=='eta': b=np.linspace(-3.5,3.5,40)
    elif obs=='phi': b=np.linspace(-3.5,3.5,40)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    hep.atlas.label(data=False, label="Internal",com=13.6)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    r.grid(True, axis='y')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f'{algo} {matching[category]} {sample}',fontsize=8)
    if obj == 'leptop': ax.set_title(f'{algo} leptonic top {sample}',fontsize=8)
    
    label = np.ones_like(top)
    if sample == 'sig': label = (y==1)
    elif sample == 'bkg': label = (y==0)

    if obj == 'top':  
        if category > 4: ax.hist(target_top,weights=1*(match==category)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(had_top_mass,weights=1*(match==category)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(top,weights=1*(match==category)*(label),histtype='step',label='Reco default (assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(max_idxs_multi_had_top_mass,weights=1*(match==category)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(had_top_mass_min,weights=1*(match==category)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
        #ax.hist(baseline_top_mass,weights=1*(match==category)*(label),histtype='step',label='Reco baseline',density=False,bins=b,color=colors[5])
        if category > 4: 
            hist_spanet = np.histogram(had_top_mass, bins=b,weights=1*(match==category)*(label))[0] 
            #hist_baseline = np.histogram(baseline_top_mass, bins=b,weights=1*(match==category)*(label))[0] 
            hist_truth = np.histogram(target_top, bins=b,weights=1*(match==category)*(label))[0] 
            r.stairs( np.nan_to_num(hist_spanet/hist_truth,posinf=0),b,linewidth=1,color=colors[1]) 
            #r.stairs( np.nan_to_num(hist_baseline/hist_truth,posinf=0),b,linewidth=1,color=colors[5]) 
        else:
            hist_spanet = np.histogram(had_top_mass, bins=b,weights=1*(match==category)*(label))[0] 
            r.stairs( np.nan_to_num(hist_spanet/hist_spanet,posinf=0),b,linewidth=0,color=colors[1]) 
    elif obj == 'W':  
        if category > 4: ax.hist(target_w,weights=1*(match==category)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(w_mass,weights=1*(match==category)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(w,weights=1*(match==category)*(label),histtype='step',label='Reco default (assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(max_idxs_multi_w_mass,weights=1*(match==category)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(w_mass_min,weights=1*(match==category)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
        #ax.hist(baseline_W_mass,weights=1*(match==category)*(label),histtype='step',label='Reco baseline',density=False,bins=b,color=colors[5])
        if category > 4: 
            hist_spanet = np.histogram(w_mass, bins=b,weights=1*(match==category)*(label))[0] 
            #hist_baseline = np.histogram(baseline_W_mass, bins=b,weights=1*(match==category)*(label))[0] 
            hist_truth = np.histogram(target_w, bins=b,weights=1*(match==category)*(label))[0] 
            r.stairs( np.nan_to_num(hist_spanet/hist_truth,posinf=0),b,linewidth=1,color=colors[1]) 
            #r.stairs( np.nan_to_num(hist_baseline/hist_truth,posinf=0),b,linewidth=1,color=colors[5]) 
        else: 
            hist_spanet = np.histogram(w_mass, bins=b,weights=1*(match==category)*(label))[0]
            r.stairs( np.nan_to_num(hist_spanet/hist_spanet,posinf=0),b,linewidth=0,color=colors[1])    
    elif obj == 'leptop':  
        ax.hist(target_ltop,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Truth matched',density=False,bins=b,color=colors[0],lw=2)
        ax.hist(lep_top_mass,weights=1*(targets_lt[:,0]!=-1)*(label),label='Reco (priority from detection prob)',density=False,bins=b, alpha=0.5,color=colors[1])
        ax.hist(ltop,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Reco default (assignment prob only)',density=False,bins=b,color=colors[2])
        ax.hist(lep_top_mass_min,weights=1*(targets_lt[:,0]!=-1)*(label),label='Reco (priority to had top)',histtype='step',density=False,bins=b,color=colors[3])
        ax.hist(max_idxs_multi_lep_top_mass,weights=1*(targets_lt[:,0]!=-1)*(label),histtype='step',label='Reco (priority to lep top)',density=False,bins=b,color=colors[4])
        if category > 4: 
            hist_spanet = np.histogram(lep_top_mass, bins=b,weights=1*(match==category)*(label))[0] 
            hist_truth = np.histogram(target_ltop, bins=b,weights=1*(match==category)*(label))[0] 
            r.stairs( np.nan_to_num(hist_spanet/hist_truth,posinf=0),b,linewidth=1,color=colors[1]) 
        else: 
            hist_spanet = np.histogram(lep_top_mass, bins=b,weights=1*(match==category)*(label))[0]
            r.stairs( np.nan_to_num(hist_spanet/hist_spanet,posinf=0),b,linewidth=0,color=colors[1])    
    else: 
        return    
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': r.set_xlabel(f'had top cand {obs} [GeV]',loc='right')
    elif obj=='W': r.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obj=='leptop': r.set_xlabel(f'lep top cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': r.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': r.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': r.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=4,loc='upper left')
    else: ax.legend(fontsize=4,loc='upper right')
    if obs in ['detection_probability','prediction_probability','prediction_probability_lt','detection_probability_lt']: 
        ax.semilogy()
    #ax.semilogy()    
    r.set_ylim(0,2)
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=8)
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
    if obj == 'leptop': plt.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}{mess}.png')
    else: plt.savefig(f'{out_dir}/{sample}_cat_{category}_{obj}_{obs}_{algo}{mess}.png')

def plot_all_categories(had_top_mass,had_top_mass_min,max_idxs_multi_had_top_mass,top,target_top,
                            w_mass,w_mass_min,max_idxs_multi_w_mass,w,target_w,
                            lep_top_mass,lep_top_mass_min,max_idxs_multi_lep_top_mass,ltop,target_ltop,
                            #baseline_top_mass,baseline_W_mass,
                            targets_lt,
                            match=match_label,out=out,y=y,sample='all',obj='top',obs='mass',algo='SPANet',thr=0.,category=5,
                           colors=[  '#1f77b4',
                                     '#ff7f0e',
                                     '#2ca02c',
                                     '#d62728',
                                     '#9467bd',
                                     '#8c564b',
                                     '#e377c2',
                                     '#7f7f7f',
                                     '#bcbd22',
                                     '#17becf'],
                                     mess=''):
    
    if obj=='top': b=np.linspace(50,400,60)
    elif obj=='leptop': b=np.linspace(0,400,60)
    elif obj=='W': b=np.linspace(0,150,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    elif obs=='eta': b=np.linspace(-3.5,3.5,40)
    elif obs=='phi': b=np.linspace(-3.5,3.5,40)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    hep.atlas.label(data=False, label="Internal",com=13.6)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f'{algo} full events {sample}',fontsize=8)
    if obj == 'leptop': ax.set_title(f'{algo} leptonic top {sample}',fontsize=8)

    label_sig = (y==1)
    label_bkg = (y==0)

    if obj == 'top':  
        ax.hist(had_top_mass,weights=1*(label_sig),histtype='step',label='Sig Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_top_mass,weights=1*(label_sig),histtype='step',label='Sig Reco baseline',density=True,bins=b,color=colors[5])

        ax.hist(had_top_mass,weights=1*(label_bkg),linestyle='dashed',histtype='step',label='Bkg Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_top_mass,weights=1*(label_bkg),linestyle='dashed',histtype='step',label='Bkg Reco baseline',density=True,bins=b,color=colors[5])

        hist_spanet = np.histogram(had_top_mass, bins=b,weights=1*(label_sig))[0] 
        #hist_baseline = np.histogram(baseline_top_mass, bins=b,weights=1*(label_sig))[0] 

        hist_spanet_bkg = np.histogram(had_top_mass, bins=b,weights=1*(label_bkg))[0] 
        #hist_baseline_bkg = np.histogram(baseline_top_mass, bins=b,weights=1*(label_bkg))[0] 

        r.stairs( np.nan_to_num(hist_spanet/hist_spanet_bkg,posinf=0),b,linewidth=1,color=colors[1]) 
        #r.stairs( np.nan_to_num(hist_baseline/hist_baseline_bkg,posinf=0),b,linewidth=1,color=colors[5]) 

    elif obj == 'W':  
        ax.hist(w_mass,weights=1*(label_sig),histtype='step',label='Sig Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_W_mass,weights=1*(label_sig),histtype='step',label='Sig Reco baseline',density=True,bins=b,color=colors[5])

        ax.hist(w_mass,weights=1*(label_bkg),linestyle='dashed',histtype='step',label='Bkg Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_W_mass,weights=1*(label_bkg),linestyle='dashed',histtype='step',label='Bkg Reco baseline',density=True,bins=b,color=colors[5])

        hist_spanet = np.histogram(w_mass, bins=b,weights=1*(label_sig))[0] 
        #hist_baseline = np.histogram(baseline_W_mass, bins=b,weights=1*(label_sig))[0] 

        hist_spanet_bkg = np.histogram(w_mass, bins=b,weights=1*(label_bkg))[0] 
        #hist_baseline_bkg = np.histogram(baseline_W_mass, bins=b,weights=1*(label_bkg))[0] 

        r.stairs( np.nan_to_num(hist_spanet/hist_spanet_bkg,posinf=0),b,linewidth=1,color=colors[1]) 
        #r.stairs( np.nan_to_num(hist_baseline/hist_baseline_bkg,posinf=0),b,linewidth=1,color=colors[5]) 

    else: 
        return    
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': r.set_xlabel(f'had top cand {obs} [GeV]',loc='right')
    elif obj=='W': r.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obj=='leptop': r.set_xlabel(f'lep top cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': r.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': r.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': r.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    if obs in ['detection_probability','prediction_probability','prediction_probability_lt','detection_probability_lt']: 
        ax.semilogy()
    #ax.semilogy()    
    r.grid(True, axis='y')
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=8)
    out_dir = f'Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/All_Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/All_Categories/{obj}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/All_Categories/{obj}/{obs}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/All_Categories/{obj}/{obs}/{sample}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    if obj == 'leptop': plt.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}{mess}.png')
    else: plt.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}{mess}.png')    

def plot_all_truth_categories(had_top_mass,had_top_mass_min,max_idxs_multi_had_top_mass,top,target_top,
                            w_mass,w_mass_min,max_idxs_multi_w_mass,w,target_w,
                            lep_top_mass,lep_top_mass_min,max_idxs_multi_lep_top_mass,ltop,target_ltop,
                            #baseline_top_mass,baseline_W_mass,
                            targets_lt,
                            match=match_label,out=out,y=y,sample='all',obj='top',obs='mass',algo='SPANet',thr=0.,category=5,
                           colors=[  '#1f77b4',
                                     '#ff7f0e',
                                     '#2ca02c',
                                     '#d62728',
                                     '#9467bd',
                                     '#8c564b',
                                     '#e377c2',
                                     '#7f7f7f',
                                     '#bcbd22',
                                     '#17becf'],
                                     mess=''):
    
    if obj=='top': b=np.linspace(50,400,60)
    elif obj=='leptop': b=np.linspace(0,400,60)
    elif obj=='W': b=np.linspace(0,150,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    elif obs=='eta': b=np.linspace(-3.5,3.5,40)
    elif obs=='phi': b=np.linspace(-3.5,3.5,40)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    hep.atlas.label(data=False, label="Internal",com=13.6)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f'{algo} full events {sample}',fontsize=8)
    if obj == 'leptop': ax.set_title(f'{algo} leptonic top {sample}',fontsize=8)

    label_sig = (y==1)*(match>4)
    label_bkg = (y==0)*(match>4)

    if obj == 'top':  
        ax.hist(target_top,weights=1*(label_sig),histtype='step',label='Sig Truth matched',density=True,bins=b,color=colors[0],lw=2)
        ax.hist(had_top_mass,weights=1*(label_sig),histtype='step',label='Sig Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_top_mass,weights=1*(label_sig),histtype='step',label='Sig Reco baseline',density=True,bins=b,color=colors[5])

        ax.hist(target_top,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Truth matched',density=True,bins=b,color=colors[0],lw=2)
        ax.hist(had_top_mass,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_top_mass,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Reco baseline',density=True,bins=b,color=colors[5])
        
        hist_spanet = np.histogram(had_top_mass, bins=b,weights=1*(label_sig))[0] 
        #hist_baseline = np.histogram(baseline_top_mass, bins=b,weights=1*(label_sig))[0] 
        hist_truth = np.histogram(target_top, bins=b,weights=1*(label_sig))[0] 

        hist_spanet_bkg = np.histogram(had_top_mass, bins=b,weights=1*(label_bkg))[0] 
        #hist_baseline_bkg = np.histogram(baseline_top_mass, bins=b,weights=1*(label_bkg))[0] 
        hist_truth_bkg = np.histogram(target_top, bins=b,weights=1*(label_bkg))[0] 

        r.stairs( np.nan_to_num(hist_spanet/hist_truth,posinf=0),b,linewidth=1,color=colors[1]) 
        #r.stairs( np.nan_to_num(hist_baseline/hist_truth,posinf=0),b,linewidth=1,color=colors[5]) 

        r.stairs( np.nan_to_num(hist_spanet_bkg/hist_truth_bkg,posinf=0),b,linewidth=1,color=colors[1],linestyle='dashed') 
        #r.stairs( np.nan_to_num(hist_baseline_bkg/hist_truth_bkg,posinf=0),b,linewidth=1,color=colors[5],linestyle='dashed') 

    elif obj == 'W':  
        ax.hist(target_w,weights=1*(label_sig),histtype='step',label='Sig Truth matched',density=True,bins=b,color=colors[0],lw=2)
        ax.hist(w_mass,weights=1*(label_sig),histtype='step',label='Sig Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_W_mass,weights=1*(label_sig),histtype='step',label='Sig Reco baseline',density=True,bins=b,color=colors[5])

        ax.hist(target_w,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Truth matched',density=True,bins=b,color=colors[0],lw=2)
        ax.hist(w_mass,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Reco (priority from detection prob)',density=True,bins=b,color=colors[1])
        #ax.hist(baseline_W_mass,weights=1*(label_bkg),histtype='step',linestyle='dashed',label='Bkg Reco baseline',density=True,bins=b,color=colors[5])

        hist_spanet = np.histogram(w_mass, bins=b,weights=1*(label_sig))[0] 
        #hist_baseline = np.histogram(baseline_W_mass, bins=b,weights=1*(label_sig))[0] 
        hist_truth = np.histogram(target_w, bins=b,weights=1*(label_sig))[0] 

        hist_spanet_bkg = np.histogram(w_mass, bins=b,weights=1*(label_bkg))[0] 
        #hist_baseline_bkg = np.histogram(baseline_W_mass, bins=b,weights=1*(label_bkg))[0] 
        hist_truth_bkg = np.histogram(target_w, bins=b,weights=1*(label_bkg))[0] 

        r.stairs( np.nan_to_num(hist_spanet/hist_truth,posinf=0),b,linewidth=1,color=colors[1]) 
        #r.stairs( np.nan_to_num(hist_baseline/hist_truth,posinf=0),b,linewidth=1,color=colors[5]) 

        r.stairs( np.nan_to_num(hist_spanet_bkg/hist_truth_bkg,posinf=0),b,linewidth=1,color=colors[1],linestyle='dashed') 
        #r.stairs( np.nan_to_num(hist_baseline_bkg/hist_truth_bkg,posinf=0),b,linewidth=1,color=colors[5],linestyle='dashed') 
    else: 
        return    
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': r.set_xlabel(f'had top cand {obs} [GeV]',loc='right')
    elif obj=='W': r.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obj=='leptop': r.set_xlabel(f'lep top cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': r.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': r.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': r.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    if obs in ['detection_probability','prediction_probability','prediction_probability_lt','detection_probability_lt']: 
        ax.semilogy()
    #ax.semilogy()    
    r.grid(True, axis='y')
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=8)
    out_dir = f'Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Truth_Categories'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Truth_Categories/{obj}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Truth_Categories/{obj}/{obs}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'Categories/Truth_Categories/{obj}/{obs}/{sample}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    if obj == 'leptop': plt.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}{mess}.png')
    else: plt.savefig(f'{out_dir}/{sample}_{obj}_{obs}_{algo}{mess}.png')   

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def get_auc(targets,predictions,title):
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets).reshape(-1),predictions.reshape(-1))
    Auc_sig = auc(fpr_sig,tpr_sig)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(fpr_sig,tpr_sig,label=f'auc : {Auc_sig:.2f}')
    plt.title(f'{title}',fontsize=8)
    hep.atlas.label(data=False, label="Internal",com=13.6)
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=12)
    ax.set_ylim(0,1)
    fig.savefig(f'{title}.pdf')

def get_auc_vs(targets_spanet,predictions_spanet,targets_base,predictions_base,title,event_weights):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_spanet),predictions_spanet,sample_weight=event_weights)
    Auc_sig_spanet = auc(fpr_sig,tpr_sig)
    ax.plot(fpr_sig,tpr_sig,label=f'spanet auc : {Auc_sig_spanet:.2f}')
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_base),predictions_base,sample_weight=event_weights)
    Auc_sig_base = auc(fpr_sig,tpr_sig)
    ax.plot(fpr_sig,tpr_sig,label=f'baseline auc : {Auc_sig_base:.2f}')
    plt.title(f'{title}',fontsize=8)
    #ax.legend()
    hep.atlas.label(data=False, label="Internal",com=13.6)
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=12)
    ax.set_ylim(0,1)
    fig.savefig(f'{title}.pdf')

def plot_cat_cut(outputs,match_label):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    b=np.linspace(-1,1,100)
    ax.hist(outputs[-1][:,-2]-outputs[-1][:,-1],weights=1*(match_label==6),histtype='step',bins=b,density=True,label='cat 6')
    ax.hist(outputs[-1][:,-2]-outputs[-1][:,-1],weights=1*(match_label==5),histtype='step',bins=b,density=True,label='cat 5')
    hep.atlas.label(data=False, label="Internal",com=13.6)
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=12)
    fig.savefig(f'cat6_vs_5.pdf')
    ax.legend()     

def get_bkg_rej_vs(targets_spanet,predictions_spanet,targets_base,predictions_base,title,event_weights):
    tpr_common = np.linspace(0,1,10000)
    plt.figure(figsize=(8, 6), dpi=600)
    ax = plt.subplot2grid((4, 4), (0, 0), rowspan=3,colspan=4)
    hep.atlas.label(data=False, label="Internal",com=13.6)
    r = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_spanet),predictions_spanet,sample_weight=event_weights)
    Auc_sig_spanet = auc(fpr_sig,tpr_sig)
    fpr_sig_spanet = np.interp(tpr_common,tpr_sig,fpr_sig)
    
    ax.plot(tpr_common,1/fpr_sig_spanet,label=f'spanet auc : {Auc_sig_spanet:.2f}')
    fpr_sig, tpr_sig, thresholds_sig = roc_curve((targets_base),predictions_base,sample_weight=event_weights)
    Auc_sig_base = auc(fpr_sig,tpr_sig)
    fpr_sig_baseline = np.interp(tpr_common,tpr_sig,fpr_sig)
    ax.plot(tpr_common,1/fpr_sig_baseline,label=f'baseline auc : {Auc_sig_base:.2f}')
    ax.set_title(f'{title}',fontsize=8)
    #ax.legend()
    ax.legend(bbox_to_anchor=(0.6, 0.85), loc="upper left",fontsize=12)
    r.plot( tpr_common,np.nan_to_num(fpr_sig_baseline/fpr_sig_spanet,posinf=0),color=colors[0]) 
    r.plot( tpr_common,np.nan_to_num(fpr_sig_baseline/fpr_sig_baseline,posinf=0),color=colors[1]) 
    ax.set_xlim(0.4,1)
    ax.set_ylim(1,100)
    r.set_ylim(0,5)
    r.set_xlim(0.4,1)
    r.set_xlabel(f'matched top efficiency',loc='right')
    ax.semilogy() 
    ax.set_ylabel(f'background rejection')
    plt.savefig(f'{title}.pdf')      

if __name__ == "__main__":

    if True:

        #get_auc_vs(y,outputs[-2][:,0],y,outputs_old[-2][:,1],'sVSb',train_weights)
        #get_bkg_rej_vs(y,outputs[-2][:,0],y,outputs_old[-2][:,1],'sVsb',train_weights)
        #get_bkg_rej_vs(np.array(labels_evt).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2])),labels_evt,max_baseline,'tagging_5_6_spanet_vs_baseline_bkg_rej',event_weights)
        #get_bkg_rej_vs(np.array(labels_evt[y==1]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==1],np.array(labels_evt[y==1]).astype(int),max_baseline[y==1],'tagging_5_6_spanet_vs_baseline_sig_bkg_rej',event_weights[y==1])
        #get_bkg_rej_vs(np.array(labels_evt[y==0]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==0],np.array(labels_evt[y==0]).astype(int),max_baseline[y==0],'tagging_5_6_spanet_vs_baseline_bkg_bkg_rej',event_weights[y==0])
        plot_cat_cut(outputs,match_label)

        #get_auc(labels,outputs_baseline[0],'baseline_auc')
        #get_auc_vs(np.array(labels_evt).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2])),labels_evt,max_baseline,'tagging_5_6_spanet_vs_baseline',event_weights)
        #get_auc_vs(np.array(labels_evt[y==1]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==1],np.array(labels_evt[y==1]).astype(int),max_baseline[y==1],'tagging_5_6_spanet_vs_baseline_sig',event_weights[y==1])
        #get_auc_vs(np.array(labels_evt[y==0]).astype(int),(outputs[-1][:,-1]+(outputs[-1][:,-2]))[y==0],np.array(labels_evt[y==0]).astype(int),max_baseline[y==0],'tagging_5_6_spanet_vs_baseline_bkg',event_weights[y==0])
    if True:
        #print('baseline accuracy on pairs : ', np.sum((np.sum(target_pt_baseline[(match_label==5)][:,:2]==(np.array(multiplets_evt[max_baseline_idx][:,:,:2]).reshape(-1,2)[(match_label==5)]),axis=-1)==2)*pair_baseline[(match_label==5)])/np.sum(match_label==5))
        #print('baseline accuracy on triplets : ', np.sum(
        #((np.sum(target_pt_baseline[(match_label==6)][:,:3]==(np.array(multiplets_evt[max_baseline_idx][:,:,:3]).reshape(-1,3)[(match_label==6)]),axis=-1)==3)
        #+(np.sum(target_pt_baseline[(match_label==6)][:,[0,2,1]]==(np.array(multiplets_evt[max_baseline_idx][:,:,:3]).reshape(-1,3)[(match_label==6)]),axis=-1)==3))
        #)/np.sum(match_label==6))
#
        had_top, lep_top, max_idxs_multi_had_top, max_idxs_multi_lep_top, had_top_min, lep_top_min = get_best(outputs)
        lep_top = np.concatenate((lep_top,np.ones(len(lep_top)).reshape(len(lep_top),-1)*7),axis=-1).astype(int)
        max_idxs_multi_lep_top = np.concatenate((max_idxs_multi_lep_top,np.ones(len(max_idxs_multi_lep_top)).reshape(len(max_idxs_multi_lep_top),-1)*7),axis=-1).astype(int)
        lep_top_min = np.concatenate((lep_top_min,np.ones(len(lep_top_min)).reshape(len(lep_top_min),-1)*7),axis=-1).astype(int)
#
        print('spanet accuracy on triplets : ', np.sum((np.sum(had_top[(match_label==6)][:,:3]==(targets[:,:3])[(match_label==6)],axis=-1)==3))/np.sum(match_label==6) )
        print('spanet accuracy on pairs : ', np.sum((np.sum(had_top[(match_label==5)][:,[0,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(had_top[(match_label==5)][:,[1,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(had_top[(match_label==5)][:,[0,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(had_top[(match_label==5)][:,[2,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(had_top[(match_label==5)][:,[1,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(had_top[(match_label==5)][:,[2,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2))/np.sum(match_label==5) )

        print('spanet had prio accuracy on triplets : ', np.sum((np.sum(max_idxs_multi_had_top[(match_label==6)][:,:3]==(targets[:,:3])[(match_label==6)],axis=-1)==3))/np.sum(match_label==6) )
        print('spanet had prio accuracy on pairs : ', np.sum((np.sum(max_idxs_multi_had_top[(match_label==5)][:,[0,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(max_idxs_multi_had_top[(match_label==5)][:,[1,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(max_idxs_multi_had_top[(match_label==5)][:,[0,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(max_idxs_multi_had_top[(match_label==5)][:,[2,0]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(max_idxs_multi_had_top[(match_label==5)][:,[1,2]]==(targets[:,:2])[(match_label==5)],axis=-1)==2)
           +(np.sum(max_idxs_multi_had_top[(match_label==5)][:,[2,1]]==(targets[:,:2])[(match_label==5)],axis=-1)==2))/np.sum(match_label==5) )   


        had_top_mass = get_observable(pt,phi,eta,mass,had_top,masks,thr=0.,reco='top',obs='mass')
        had_top_mass_min = get_observable(pt,phi,eta,mass,had_top_min,masks,thr=0.,reco='top',obs='mass')
        max_idxs_multi_had_top_mass = get_observable(pt,phi,eta,mass,max_idxs_multi_had_top,masks,thr=0.,reco='top',obs='mass')
        top = get_observable(pt,phi,eta,mass,out[1],masks,thr=0.,reco='top',obs='mass')
        target_top = get_observable(pt,phi,eta,mass,targets,masks,thr=0.,reco='top',obs='mass')
        #baseline_top_mass = get_observable_baseline(baseline_preds,reco='top',obs='mass')


        w_mass = get_observable(pt,phi,eta,mass,had_top,masks,thr=0.,reco='W',obs='mass')
        w_mass_min = get_observable(pt,phi,eta,mass,had_top_min,masks,thr=0.,reco='W',obs='mass')
        max_idxs_multi_w_mass = get_observable(pt,phi,eta,mass,max_idxs_multi_had_top,masks,thr=0.,reco='W',obs='mass')
        w = get_observable(pt,phi,eta,mass,out[1],masks,thr=0.,reco='W',obs='mass')
        target_w = get_observable(pt,phi,eta,mass,targets,masks,thr=0.,reco='W',obs='mass')
        #baseline_W_mass = get_observable_baseline(baseline_preds,reco='W',obs='mass')

        lep_top_mass = get_observable_leptop(pt,phi,eta,mass,lep_top,masks,thr=0.,reco='top',obs='mass')
        lep_top_mass_min = get_observable_leptop(pt,phi,eta,mass,lep_top_min,masks,thr=0.,reco='top',obs='mass')
        max_idxs_multi_lep_top_mass = get_observable_leptop(pt,phi,eta,mass,max_idxs_multi_lep_top,masks,thr=0.,reco='top',obs='mass')
        ltop = get_observable_leptop(pt,phi,eta,mass,np.concatenate((out[0],np.ones(len(out[0])).reshape(len(out[0]),-1)*7),axis=-1).astype(int),masks,thr=0.,reco='top',obs='mass')
        target_ltop = get_observable_leptop(pt,phi,eta,mass,targets_lt,masks,thr=0.,reco='top',obs='mass')


        had_top_mass_0 = get_observable(pt,phi,eta,mass,had_top,masks,cut_prob=(outputs[-1][:,-2]-(outputs[-1][:,-1])),thr=0.,reco='top',obs='mass')
        w_mass_0 = get_observable(pt,phi,eta,mass,had_top,masks,cut_prob=(outputs[-1][:,-2]-(outputs[-1][:,-1])),thr=0.,reco='W',obs='mass')

        had_top_mass_1 = get_observable(pt,phi,eta,mass,had_top,masks,cut_prob=(outputs[-1][:,-2]-(outputs[-1][:,-1])),thr=-1.,reco='top',obs='mass')
        w_mass_1 = get_observable(pt,phi,eta,mass,had_top,masks,cut_prob=(outputs[-1][:,-2]-(outputs[-1][:,-1])),thr=-1.,reco='W',obs='mass')

    #if True:
        #for obj in ['top','W']:
        #    plot_all_categories(had_top_mass.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
        #                                   w_mass.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
        #                                   lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
        #                                   #baseline_top_mass.mass,baseline_W_mass.mass,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_categories(had_top_mass.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
        #                                   w_mass.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
        #                                   lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
        #                                   #baseline_top_mass.pt,baseline_W_mass.pt,
        #                                   
        #                            targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_categories(had_top_mass.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
        #                                   w_mass.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
        #                                   lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
        #                                   #baseline_top_mass.eta,baseline_W_mass.eta,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_categories(had_top_mass.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
        #                                   w_mass.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
        #                                   lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
        #                                   #baseline_top_mass.phi,baseline_W_mass.phi,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_truth_categories(had_top_mass.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
        #                                   w_mass.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
        #                                   lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
        #                                   #baseline_top_mass.mass,baseline_W_mass.mass,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_truth_categories(had_top_mass.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
        #                                   w_mass.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
        #                                   lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
        #                                   #baseline_top_mass.pt,baseline_W_mass.pt,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_truth_categories(had_top_mass.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
        #                                   w_mass.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
        #                                   lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
        #                                   #baseline_top_mass.eta,baseline_W_mass.eta,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_truth_categories(had_top_mass.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
        #                                   w_mass.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
        #                                   lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
        #                                   #baseline_top_mass.phi,baseline_W_mass.phi,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=6,colors=colors) 
        #    plot_all_truth_categories(had_top_mass_0.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
        #                                   w_mass_0.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
        #                                   lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
        #                                   #baseline_top_mass.mass,baseline_W_mass.mass,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=6,colors=colors,mess='_cut_0') 
        #    plot_all_truth_categories(had_top_mass_0.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
        #                                   w_mass_0.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
        #                                   lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
        #                                   #baseline_top_mass.pt,baseline_W_mass.pt,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=6,colors=colors,mess='_cut_0') 
        #    plot_all_truth_categories(had_top_mass_0.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
        #                                   w_mass_0.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
        #                                   lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
        #                                   #baseline_top_mass.eta,baseline_W_mass.eta,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=6,colors=colors,mess='_cut_0') 
        #    plot_all_truth_categories(had_top_mass_0.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
        #                                   w_mass_0.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
        #                                   lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
        #                                   #baseline_top_mass.phi,baseline_W_mass.phi,
        #                                   targets_lt,
        #                                    sample='all',out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=6,colors=colors,mess='_cut_0') 

        #for sample in ['sig']:
        #    for category in [6,3,5]:
        #        for obj in ['top','W']:
        #            if (obj=='leptop' and category!=6): continue
        #            plot_single_categories(had_top_mass_0.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
        #                                   w_mass_0.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
        #                                   lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
        #                                   #baseline_top_mass.mass,baseline_W_mass.mass,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_0')  
        #            plot_single_categories(had_top_mass_1.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
        #                                   w_mass_1.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
        #                                   lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
        #                                   #baseline_top_mass.mass,baseline_W_mass.mass,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_all')  
        #            plot_single_categories(had_top_mass_0.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
        #                                   w_mass_0.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
        #                                   lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
        #                                   #baseline_top_mass.pt,baseline_W_mass.pt,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_0')  
        #            plot_single_categories(had_top_mass_1.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
        #                                   w_mass_1.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
        #                                   lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
        #                                   #baseline_top_mass.pt,baseline_W_mass.pt,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_all')  
        #            plot_single_categories(had_top_mass_0.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
        #                                   w_mass_0.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
        #                                   lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
        #                                   #baseline_top_mass.eta,baseline_W_mass.eta,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_0')  
        #            plot_single_categories(had_top_mass_1.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
        #                                   w_mass_1.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
        #                                   lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
        #                                   #baseline_top_mass.eta,baseline_W_mass.eta,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_all')  
        #            plot_single_categories(had_top_mass_0.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
        #                                   w_mass_0.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
        #                                   lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
        #                                   #baseline_top_mass.phi,baseline_W_mass.phi,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_0')  
        #            plot_single_categories(had_top_mass_1.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
        #                                   w_mass_1.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
        #                                   lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
        #                                   #baseline_top_mass.phi,baseline_W_mass.phi,
        #                                   targets_lt,
        #                                    sample=sample,out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=category,colors=colors,mess='_cut_all')  

    if True:    
        for sample in ['sig']:
            for category in [6,3,0,1,2,4,5]:
                for obj in ['top','W','leptop']:
                    if (obj=='leptop' and category!=6): continue
                    plot_single_categories(had_top_mass.mass,had_top_mass_min.mass,max_idxs_multi_had_top_mass.mass,top.mass,target_top.mass,
                                               w_mass.mass,w_mass_min.mass,max_idxs_multi_w_mass.mass,w.mass,target_w.mass,
                                               lep_top_mass.mass,lep_top_mass_min.mass,max_idxs_multi_lep_top_mass.mass,ltop.mass,target_ltop.mass,
                                               #baseline_top_mass.mass,baseline_W_mass.mass,
                                               targets_lt,
                                                sample=sample,out=out,y=y,obj=obj,obs='mass',algo='SPANet',thr=0,category=category,colors=colors) 
                    plot_single_categories(had_top_mass.pt,had_top_mass_min.pt,max_idxs_multi_had_top_mass.pt,top.pt,target_top.pt,
                                               w_mass.pt,w_mass_min.pt,max_idxs_multi_w_mass.pt,w.pt,target_w.pt,
                                               lep_top_mass.pt,lep_top_mass_min.pt,max_idxs_multi_lep_top_mass.pt,ltop.pt,target_ltop.pt,
                                               #baseline_top_mass.pt,baseline_W_mass.pt,
                                               targets_lt,
                                                sample=sample,out=out,y=y,obj=obj,obs='pt',algo='SPANet',thr=0,category=category,colors=colors) 
                    plot_single_categories(had_top_mass.eta,had_top_mass_min.eta,max_idxs_multi_had_top_mass.eta,top.eta,target_top.eta,
                                               w_mass.eta,w_mass_min.eta,max_idxs_multi_w_mass.eta,w.eta,target_w.eta,
                                               lep_top_mass.eta,lep_top_mass_min.eta,max_idxs_multi_lep_top_mass.eta,ltop.eta,target_ltop.eta,
                                               #baseline_top_mass.eta,baseline_W_mass.eta,
                                               targets_lt,
                                                sample=sample,out=out,y=y,obj=obj,obs='eta',algo='SPANet',thr=0,category=category,colors=colors) 
                    plot_single_categories(had_top_mass.phi,had_top_mass_min.phi,max_idxs_multi_had_top_mass.phi,top.phi,target_top.phi,
                                               w_mass.phi,w_mass_min.phi,max_idxs_multi_w_mass.phi,w.phi,target_w.phi,
                                               lep_top_mass.phi,lep_top_mass_min.phi,max_idxs_multi_lep_top_mass.phi,ltop.phi,target_ltop.phi,
                                               #baseline_top_mass.phi,baseline_W_mass.phi,
                                               targets_lt,
                                                sample=sample,out=out,y=y,obj=obj,obs='phi',algo='SPANet',thr=0,category=category,colors=colors) 

   