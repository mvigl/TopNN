import numpy as np
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import torch
from sklearn.metrics import roc_curve,auc
import math
import vector 
import os
import yaml
import h5py
import pickle

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
device = get_device()
device = 'cpu'

def make_mlp(in_features,out_features,nlayer,for_inference=False,binary=True):
    layers = []
    for i in range(nlayer):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    if binary: layers.append(torch.nn.Linear(in_features, 1))
    if for_inference: layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    return model

def load_weights(model,weights,device):
    pretrained_dict = torch.load(weights,map_location=torch.device(device))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print('loading weights :')
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    return model

def get_scores(weights,data,device,in_features=12,out_features=128,nlayer=4,for_inference=True,binary=True):
    eval_model = make_mlp(in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=for_inference,binary=binary)
    eval_model = load_weights(eval_model,weights,device)
    with torch.no_grad():
        eval_model.eval()
        eval_model.to(device)
        preds = eval_model(data)
    return preds.detach().cpu().numpy()

def get_AUC_topMAX(results,model,sample='stop',label='top_Maxscore_label'):
    preds_sig = results[sample][model]['evt']['top_Maxscore']
    label = results[sample][model]['evt'][label]
    fpr_sig, tpr_sig, thresholds_sig = roc_curve(label,preds_sig)
    Auc_sig = auc(fpr_sig,tpr_sig)
    return Auc_sig

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
#5 for hadronic decaying top with matched bjet and 2 matched light jets
#4 for hadronic decaying top with matched bjet and 1 matched light jets
#3 for hadronic decaying top with matched bjet
#2 for hadronic decaying top with no matched bjet and 2 matched light jets
#1 for hadronic decaying top with no matched bjet and 1 matched light jet
#0 for hadronic decaying top if matching procedure failed
#-1 for leptonic decaying top or if topp(topn) doesn't exist


def get_inputs(file,samples):
    
    with h5py.File(file, 'r') as f:
        i=0
        for name in samples:
            print('reading : ',name)
            length = len(f[name]['labels'])
            if i ==0:
                truth_info = f[name]['variables']
                data = f[name]['multiplets']
                target = f[name]['labels']
            else:
                truth_info_i = f[name]['variables']
                data_i = f[name]['multiplets']
                target_i = f[name]['labels']
                data = np.concatenate((data,data_i),axis=0)
                target = np.concatenate((target,target_i),axis=0)
                truth_info = np.concatenate((truth_info,truth_info_i),axis=0)
            i+=1

        x = torch.from_numpy(data).float().to(device)    
        y = target.reshape(-1,1)
        return x,y,truth_info
    
def get_observable(top_Maxinputs,inputs,reco='top',obs='mass'):
    b= vector.array(
        {
            "pt": top_Maxinputs[:,inputs.index('bjet_pT')],
            "phi": top_Maxinputs[:,inputs.index('bjet_phi')],
            "eta": top_Maxinputs[:,inputs.index('bjet_eta')],
            "M": top_Maxinputs[:,inputs.index('bjet_M')],
        }
    )
    j1 = vector.array(
        {
            "pt": top_Maxinputs[:,inputs.index('jet1_pT')],
            "phi": top_Maxinputs[:,inputs.index('jet1_phi')],
            "eta": top_Maxinputs[:,inputs.index('jet1_eta')],
            "M": top_Maxinputs[:,inputs.index('jet1_M')],
        }
    )
    j2 = vector.array(
        {
            "pt": top_Maxinputs[:,inputs.index('jet2_pT')],
            "phi": top_Maxinputs[:,inputs.index('jet2_phi')],
            "eta": top_Maxinputs[:,inputs.index('jet2_eta')],
            "M": top_Maxinputs[:,inputs.index('jet2_M')],
        }
    )
    if reco == 'top': obj = b+(j1+j2)
    elif reco == 'W': obj = (j1+j2)
    else: 
        print('choose reco: top, W')
        return 0
    if obs=='mass': observable = obj.mass
    else: 
        print('choose observable: mass')
        return 0
    return observable

def get_results(file,samples):

    x,y,truth_info = get_inputs(file,samples)
    weights = '/u/mvigl/Stop/run/Final/Stop_FS_nodes128_layers4_lr0.0001_bs512_1000000.pt'

    in_features = 12
    out_features = int(weights[weights.index("nodes")+5:weights.index("_layers")])
    nlayer = int(weights[weights.index("layers")+6:weights.index("_lr")])
    preds = get_scores(weights,x,device,in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
    top_Maxinputs = np.array(ak.unflatten(x.detach().cpu().numpy(), np.array(truth_info[:,0]).astype(int))[ak.argmax(ak.unflatten(preds.reshape(-1), np.array(truth_info[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
    top_mass = get_observable(top_Maxinputs,inputs,reco='top',obs='mass')
    return top_mass

if __name__ == "__main__":
    
    file = '/raven/u/mvigl/Stop/data/H5_full/Virtual_test.h5'
    with open('/raven/u/mvigl/Stop/TopNN/data/H5/test/list_all.txt', "r") as filter:
        samples = [line.strip() for line in filter.readlines()]

    results = get_results(file,samples)
    dir_out = 'results'
    if (not os.path.exists(dir_out)): os.system(f'mkdir {dir_out}')
    with open(f'{dir_out}/baseline_results.h5', 'wb') as f:
        pickle.dump(results, f) 
