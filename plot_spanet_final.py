import h5py
import vector
import os
import onnxruntime
import vector 
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', help='data',default='data/root/list_sig_FS_testing.txt')
parser.add_argument('--evals', help='evals',default='data/root/list_sig_FS_testing.txt')
args = parser.parse_args()

from typing import List
import numba
from numba import njit

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
    samples = np.arange(len(h5fw['INPUTS']['Momenta']['pt'][:]))
    np.random.shuffle(samples)
    samples = samples[:10000]
    pt = h5fw['INPUTS']['Momenta']['pt'][:][samples]
    eta = h5fw['INPUTS']['Momenta']['eta'][:][samples]
    phi = h5fw['INPUTS']['Momenta']['phi'][:][samples]
    mass = h5fw['INPUTS']['Momenta']['mass'][:][samples]
    masks = h5fw['INPUTS']['Momenta']['MASK'][:][samples]
    targets = np.column_stack((h5fw['TARGETS']['ht']['b'][:][samples],h5fw['TARGETS']['ht']['q1'][:][samples],h5fw['TARGETS']['ht']['q2'][:][samples]))
    targets_lt = h5fw['TARGETS']['lt']['b'][:][samples]
    targets_lt = targets_lt.reshape((len(targets_lt),-1))
    targets_lt = np.concatenate((targets_lt,np.ones(len(targets_lt)).reshape(len(targets_lt),-1)*7),axis=-1).astype(int)
    match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][:][samples]
    nbs = h5fw['truth_info']['nbjet'][:][samples]
    is_matched = h5fw['truth_info']['is_matched'][:][samples]
     
    Momenta_data = np.array([h5fw['INPUTS']['Momenta']['mass'][:][samples],
                    h5fw['INPUTS']['Momenta']['pt'][:][samples],
                    h5fw['INPUTS']['Momenta']['eta'][:][samples],
                    h5fw['INPUTS']['Momenta']['phi'][:][samples],
                    h5fw['INPUTS']['Momenta']['btag'][:][samples],
                    h5fw['INPUTS']['Momenta']['qtag'][:][samples],
                    h5fw['INPUTS']['Momenta']['etag'][:][samples]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(h5fw['INPUTS']['Momenta']['MASK'][:][samples]).astype(bool)

    Met_data = np.array([h5fw['INPUTS']['Met']['MET'][:][samples],
                    h5fw['INPUTS']['Met']['METsig'][:][samples],
                    h5fw['INPUTS']['Met']['METphi'][:][samples],
                    h5fw['INPUTS']['Met']['MET_Soft'][:][samples],
                    h5fw['INPUTS']['Met']['MET_Jet'][:][samples],
                    h5fw['INPUTS']['Met']['MET_Ele'][:][samples],
                    h5fw['INPUTS']['Met']['MET_Muon'][:][samples],
                    h5fw['INPUTS']['Met']['mT_METl'][:][samples],
                    h5fw['INPUTS']['Met']['dR_bb'][:][samples],
                    h5fw['INPUTS']['Met']['dphi_METl'][:][samples],
                    h5fw['INPUTS']['Met']['MT2_bb'][:][samples],
                    h5fw['INPUTS']['Met']['MT2_b1l1_b2'][:][samples],
                    h5fw['INPUTS']['Met']['MT2_b2l1_b1'][:][samples],
                    h5fw['INPUTS']['Met']['MT2_min'][:][samples],
                    h5fw['INPUTS']['Met']['HT'][:][samples],
                    h5fw['INPUTS']['Met']['nbjet'][:][samples],
                    h5fw['INPUTS']['Met']['nljet'][:][samples],
                    h5fw['INPUTS']['Met']['nVx'][:][samples]]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

    y = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:][samples]
print('Momenta_data : ', Momenta_data.shape)  
print('Momenta_mask : ', Momenta_mask.shape)  
print('Met_data : ', Met_data.shape)    
print('Met_mask : ', Met_mask.shape)   

inputs = {}
inputs['Momenta_data'] = Momenta_data
inputs['Momenta_mask'] = Momenta_mask
inputs['Met_data'] = Met_data
inputs['Met_mask'] = Met_mask

inputs.keys()

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

ort_sess = ort.InferenceSession("/raven/u/mvigl/TopReco/SPANet/spanet_log_norm.onnx")

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

def plot_single_categories(match=match_label,out=out,y=y,sample='sig',obj='top',obs='mass',algo='SPANet',thr=0.,category=5,
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
    
    if obj=='top': b=np.linspace(0,400,40)
    elif obj=='W': b=np.linspace(0,140,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plt.title(f'{algo} {matching[category]} {sample}')
    
    label = np.ones_like(top)
    if sample == 'sig': label = (y==1)
    elif sample == 'bkg': label = (y==0)

    if obj == 'top':  
        ax.hist([     
                    target_top,
                    had_top_mass,
                    top,
                    max_idxs_multi_had_top_mass,
                    had_top_mass_min,
                    ],
                    bins=b,
                    alpha=0.8,
                    weights=[
                        1*(match==category)*(label),
                        1*(match==category)*(label),
                        1*(match==category)*(label),
                        1*(match==category)*(label),
                        1*(match==category)*(label),
                    ],
                    stacked=False,
                    color=colors[:7],
                    label=[
                        f'Truth matched',
                        f'Reco (priority from detection prob)',
                        f'Default algo (based on assignment prob)',
                        f'Reco (priority to had top)',
                        f'Reco (priority to lep top)',
                    ],    
                )
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
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
    fig.savefig(f'{out_dir}/{sample}_cat_{category}_{obj}_{obs}_{algo}.png')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if __name__ == "__main__":
    
    had_top, lep_top, max_idxs_multi_had_top, max_idxs_multi_lep_top, had_top_min, lep_top_min = get_best(outputs)
    lep_top = np.concatenate((lep_top,np.ones(len(lep_top)).reshape(len(lep_top),-1)*7),axis=-1).astype(int)
    max_idxs_multi_lep_top = np.concatenate((max_idxs_multi_lep_top,np.ones(len(max_idxs_multi_lep_top)).reshape(len(max_idxs_multi_lep_top),-1)*7),axis=-1).astype(int)
    lep_top_min = np.concatenate((lep_top_min,np.ones(len(lep_top_min)).reshape(len(lep_top_min),-1)*7),axis=-1).astype(int)

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

    for sample in ['all','sig','bkg']:
        for category in [6,3,0,1,2,4,5]:
            for obj in ['top','W']:
                for obs in ['mass','pt']:
                    if (obj=='W' and obs=='pt'): continue
                    plot_single_categories(had_top_mass,had_top_mass_min,max_idxs_multi_had_top_mass,top,target_top,
                                           w_mass,w_mass_min,max_idxs_multi_w_mass,w,target_w,
                                            sample=sample,out=out,y=y,obj=obj,obs=obs,algo='SPANet',thr=0,category=category,colors=colors)  

   