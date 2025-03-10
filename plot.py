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

models_name = {
 'Stop_FS_10000':'Trained on Stop-FS (2.714.253)',
 'Stop_FS_50000':'Trained on Stop-FS (4.440.130)',
 'Stop_FS_100000':'Trained on Stop-FS (5.217.010)',
 'Stop_FS_1000000': 'Trained on Stop-FS', #5.830.734
 'Full_bkg_65000': 'Trained on Background (5.523.543)',
 'Full_bkg_68000': 'Trained on Background (5.772.543)',
 'Full_bkg_70000': 'Trained on Background (5.938.543)',  
 'Full_bkg_80000': 'Trained on Background (6.768.543)',
 'Full_bkg_100000': 'Trained on Background (8.400.280)',
 'Full_bkg_200000': 'Trained on Background (16.082.372)',
 'Full_bkg_1000000': 'Trained on Background (70.404.619)',
 'Slicing_Full_bkg_1000000': 'Trained on Background',
 'Slicing_Full_1000000': 'Trained on Stop-FS and Background (#evt)',
 'Slicing_Full_200000': 'Trained on Stop-FS and Background',
}

def get_inputs(file,samples,idmap):
    with open(idmap) as m:
        map = yaml.load(m, Loader=yaml.FullLoader)['samples'] 
    data_signals = {}
    with h5py.File(file, 'r') as f:
        i=0
        for name in samples:
            number = name[(name.index("TeV.")+4):(name.index(".stop1L"))]
            if number not in map.keys(): 
                print('skip')
                continue
            print('reading : ',name)
            length = len(f[name]['labels'])
            sample = map[number]
            if i ==0:
                truth_info = f[name]['variables']
                data = f[name]['multiplets']
                target = f[name]['labels']
                if '_Signal_' in name:
                    data_signals[sample] = {
                            'x': data,
                            'y': target,
                            'truth_info': truth_info,
                        }   
            else:
                truth_info_i = f[name]['variables']
                data_i = f[name]['multiplets']
                target_i = f[name]['labels']
                data = np.concatenate((data,data_i),axis=0)
                target = np.concatenate((target,target_i),axis=0)
                truth_info = np.concatenate((truth_info,truth_info_i),axis=0)
                if '_Signal_' in name:
                    if sample not in data_signals.keys():
                        data_signals[sample] = {
                            'x': data_i,
                            'y': target_i,
                            'truth_info': truth_info_i,
                        }
                    else:
                         data_signals[sample]['x'] = np.concatenate((data_signals[sample]['x'],data_i),axis=0)
                         data_signals[sample]['y'] = np.concatenate((data_signals[sample]['y'],target_i),axis=0) 
                         data_signals[sample]['truth_info'] = np.concatenate((data_signals[sample]['truth_info'],truth_info_i),axis=0) 
            i+=1

        x = torch.from_numpy(data).float().to(device)    
        y = target.reshape(-1,1)
        for sample in data_signals.keys():
            data_signals[sample]['x'] = torch.from_numpy(data_signals[sample]['x']).float().to(device)   
            data_signals[sample]['y'] = data_signals[sample]['y'].reshape(-1,1)
        return x,y,truth_info,data_signals
    
def get_results(file,samples_sig,samples_bkg,idmap,models=None):

    x,y,truth_info,data_signals = get_inputs(file,samples_sig,idmap)
    x_bkg,y_bkg,truth_info_bkg,empty = get_inputs(file,samples_bkg,idmap)

    if models is None:
        models = [  '/u/mvigl/Stop/run/Final/Stop_FS_nodes128_layers4_lr0.0001_bs512_10000.pt',
                    '/u/mvigl/Stop/run/Final/Stop_FS_nodes128_layers4_lr0.0001_bs512_50000.pt',
                    '/u/mvigl/Stop/run/Final/Stop_FS_nodes128_layers4_lr0.0001_bs512_100000.pt',
                    '/u/mvigl/Stop/run/Final/Stop_FS_nodes128_layers4_lr0.0001_bs512_1000000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_65000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_68000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_70000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_80000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_100000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_200000.pt',
                    '/u/mvigl/Stop/run/Final/Full_bkg_nodes128_layers4_lr0.0001_bs512_1000000.pt',
                    '/u/mvigl/Stop/run/Final/Slicing_Full_bkg_nodes128_layers4_lr0.0001_bs512_1000000.pt',
                    '/u/mvigl/Stop/run/Final/Slicing_Full_nodes128_layers4_lr0.0001_bs512_200000.pt',
                    '/u/mvigl/Stop/run/Final/Slicing_Full_nodes128_layers4_lr0.0001_bs512_1000000.pt',
          ]
          
    results = {
        'stop': {},
        'bkg': {},
        'all': {},
        'stop_samples': {},
    }    

    results['stop']['labels'] = y.reshape(-1)
    results['bkg']['labels'] = y_bkg.reshape(-1)
    y_all = np.concatenate((y,y_bkg),axis=0)
    results['all']['labels'] = y_all.reshape(-1)
    x_all = torch.cat((x,x_bkg),axis=0)
    truth_info_all = np.concatenate((truth_info,truth_info_bkg),axis=0)
    tpr_common = np.linspace(0,1,1000)
    for weights in models:
        in_features = 12
        out_features = int(weights[weights.index("nodes")+5:weights.index("_layers")])
        nlayer = int(weights[weights.index("layers")+6:weights.index("_lr")])
        preds_sig = get_scores(weights,x,device,in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
        preds_bkg = get_scores(weights,x_bkg,device,in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
        preds_all = get_scores(weights,x_all,device,in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
        name = weights[(weights.index("Final/")+6):weights.index("_nodes")] + weights[(weights.index("bs512")+5):weights.index(".pt")]

        results['stop'][name]={}
        results['stop_samples'][name]={}
        results['bkg'][name]={}
        results['all'][name]={}

        results['stop'][name]['preds'] = preds_sig
        results['bkg'][name]['preds'] = preds_bkg
        results['all'][name]['preds'] = preds_all

        fpr_sig, tpr_sig, thresholds_sig = roc_curve(results['stop']['labels'],preds_sig)
        Auc_sig = auc(fpr_sig,tpr_sig)
        fpr_sig = np.interp(tpr_common,tpr_sig,fpr_sig)
        fpr_bkg, tpr_bkg, thresholds_bkg = roc_curve(results['bkg']['labels'],preds_bkg)
        Auc_bkg = auc(fpr_bkg,tpr_bkg)
        fpr_bkg = np.interp(tpr_common,tpr_bkg,fpr_bkg)
        fpr_all, tpr_all, thresholds_all = roc_curve(results['all']['labels'],preds_all)
        Auc_all = auc(fpr_all,tpr_all)
        fpr_all = np.interp(tpr_common,tpr_all,fpr_all)

        results['stop'][name]['auc'] = Auc_sig
        results['bkg'][name]['auc'] = Auc_bkg
        results['all'][name]['auc'] = Auc_all

        results['stop'][name]['fpr'] = fpr_sig
        results['bkg'][name]['fpr'] = fpr_bkg
        results['all'][name]['fpr'] = fpr_all

        results['stop'][name]['tpr'] = tpr_common
        results['bkg'][name]['tpr'] = tpr_common
        results['all'][name]['tpr'] = tpr_common

        results['stop'][name]['evt'] = {}
        results['bkg'][name]['evt'] = {}
        results['all'][name]['evt'] = {}
        results['stop'][name]['evt']['variables'] = np.array(truth_info)
        results['bkg'][name]['evt']['variables'] = np.array(truth_info_bkg)
        results['all'][name]['evt']['variables'] = np.array(truth_info_all)

        results['stop'][name]['evt']['top_Maxscore'] = np.array(ak.max(ak.unflatten(preds_sig.reshape(-1), np.array(truth_info[:,0]).astype(int)),axis=-1)).reshape(-1)
        results['bkg'][name]['evt']['top_Maxscore'] = np.array(ak.max(ak.unflatten(preds_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int)),axis=-1)).reshape(-1)
        results['all'][name]['evt']['top_Maxscore'] = np.array(ak.max(ak.unflatten(preds_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int)),axis=-1)).reshape(-1)

        results['stop'][name]['evt']['top_Maxscore_label'] = np.array(ak.unflatten(y.reshape(-1), np.array(truth_info[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_sig.reshape(-1), np.array(truth_info[:,0]).astype(int)),keepdims=True,axis=-1)]).reshape(-1).astype(int)
        results['bkg'][name]['evt']['top_Maxscore_label'] = np.array(ak.unflatten(y_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int)),keepdims=True,axis=-1)]).reshape(-1).astype(int)
        results['all'][name]['evt']['top_Maxscore_label'] = np.array(ak.unflatten(y_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int)),keepdims=True,axis=-1)]).reshape(-1).astype(int)

        results['stop'][name]['evt']['top_Maxinputs'] = np.array(ak.unflatten(x.detach().cpu().numpy(), np.array(truth_info[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_sig.reshape(-1), np.array(truth_info[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
        results['bkg'][name]['evt']['top_Maxinputs'] = np.array(ak.unflatten(x_bkg.detach().cpu().numpy(), np.array(truth_info_bkg[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
        results['all'][name]['evt']['top_Maxinputs'] = np.array(ak.unflatten(x_all.detach().cpu().numpy(), np.array(truth_info_all[:,0]).astype(int))[ak.argmax(ak.unflatten(preds_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])

        results['stop'][name]['evt']['top_MaxisPair'] = np.array((results['stop'][name]['evt']['top_Maxinputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
        results['bkg'][name]['evt']['top_MaxisPair'] = np.array((results['bkg'][name]['evt']['top_Maxinputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
        results['all'][name]['evt']['top_MaxisPair'] = np.array((results['all'][name]['evt']['top_Maxinputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)

        results['stop'][name]['evt']['top_true_inputs'] = np.array(ak.unflatten(x.detach().cpu().numpy(), np.array(truth_info[:,0]).astype(int))[ak.argmax(ak.unflatten(y.reshape(-1), np.array(truth_info[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
        results['bkg'][name]['evt']['top_true_inputs'] = np.array(ak.unflatten(x_bkg.detach().cpu().numpy(), np.array(truth_info_bkg[:,0]).astype(int))[ak.argmax(ak.unflatten(y_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
        results['all'][name]['evt']['top_true_inputs'] = np.array(ak.unflatten(x_all.detach().cpu().numpy(), np.array(truth_info_all[:,0]).astype(int))[ak.argmax(ak.unflatten(y_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int)),keepdims=True,axis=-1)][:,0])

        results['stop'][name]['evt']['top_true_isPair'] = np.array((results['stop'][name]['evt']['top_true_inputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
        results['bkg'][name]['evt']['top_true_isPair'] = np.array((results['bkg'][name]['evt']['top_true_inputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
        results['all'][name]['evt']['top_true_isPair'] = np.array((results['all'][name]['evt']['top_true_inputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)

        results['stop'][name]['evt']['top_Matched'] = np.array(ak.max(ak.unflatten(y.reshape(-1), np.array(truth_info[:,0]).astype(int)),axis=-1)).reshape(-1).astype(int)
        results['bkg'][name]['evt']['top_Matched'] = np.array(ak.max(ak.unflatten(y_bkg.reshape(-1), np.array(truth_info_bkg[:,0]).astype(int)),axis=-1)).reshape(-1).astype(int)
        results['all'][name]['evt']['top_Matched'] = np.array(ak.max(ak.unflatten(y_all.reshape(-1), np.array(truth_info_all[:,0]).astype(int)),axis=-1)).reshape(-1).astype(int)

        acc_sig = np.sum(results['stop'][name]['evt']['top_Maxscore_label'])/np.sum(results['stop'][name]['evt']['top_Matched'])
        acc_bkg = np.sum(results['bkg'][name]['evt']['top_Maxscore_label'])/np.sum(results['bkg'][name]['evt']['top_Matched'])
        acc_all = np.sum(results['all'][name]['evt']['top_Maxscore_label'])/np.sum(results['all'][name]['evt']['top_Matched'])
        results['stop'][name]['acc'] = acc_sig
        results['bkg'][name]['acc'] = acc_bkg
        results['all'][name]['acc'] = acc_all

        acc_sig_triplets = np.sum((results['stop'][name]['evt']['top_Maxscore_label'])[results['stop'][name]['evt']['top_true_isPair']==0])/np.sum((results['stop'][name]['evt']['top_Matched'])[results['stop'][name]['evt']['top_true_isPair']==0])
        acc_bkg_triplets = np.sum((results['bkg'][name]['evt']['top_Maxscore_label'])[results['bkg'][name]['evt']['top_true_isPair']==0])/np.sum((results['bkg'][name]['evt']['top_Matched'])[results['bkg'][name]['evt']['top_true_isPair']==0])
        acc_all_triplets = np.sum((results['all'][name]['evt']['top_Maxscore_label'])[results['all'][name]['evt']['top_true_isPair']==0])/np.sum((results['all'][name]['evt']['top_Matched'])[results['all'][name]['evt']['top_true_isPair']==0])
        results['stop'][name]['acc_triplets'] = acc_sig_triplets
        results['bkg'][name]['acc_triplets'] = acc_bkg_triplets
        results['all'][name]['acc_triplets'] = acc_all_triplets

        for sample in data_signals.keys():
            preds_sig = get_scores(weights,data_signals[sample]['x'],device,in_features=12,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
            results['stop_samples'][name][sample]={}
            results['stop_samples'][name][sample]['labels'] = data_signals[sample]['y'].reshape(-1)
            results['stop_samples'][name][sample]['preds'] = preds_sig
            fpr_sig, tpr_sig, thresholds_sig = roc_curve(results['stop_samples'][name][sample]['labels'],preds_sig)
            Auc_sig = auc(fpr_sig,tpr_sig)
            fpr_sig = np.interp(tpr_common,tpr_sig,fpr_sig)
            results['stop_samples'][name][sample]['auc'] = Auc_sig
            results['stop_samples'][name][sample]['fpr'] = fpr_sig
            results['stop_samples'][name][sample]['tpr'] = tpr_common

            results['stop_samples'][name][sample]['evt'] = {}
            results['stop_samples'][name][sample]['evt']['variables'] = data_signals[sample]['truth_info']
            results['stop_samples'][name][sample]['evt']['top_Maxscore'] = np.array(ak.max(ak.unflatten(preds_sig.reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int)),axis=-1)).reshape(-1)
            results['stop_samples'][name][sample]['evt']['top_Maxscore_label'] = np.array(ak.unflatten(results['stop_samples'][name][sample]['labels'].reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int))[ak.argmax(ak.unflatten(preds_sig.reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int)),keepdims=True,axis=-1)]).reshape(-1).astype(int)
            results['stop_samples'][name][sample]['evt']['top_Maxinputs'] = np.array(ak.unflatten(data_signals[sample]['x'].detach().cpu().numpy(), np.array(data_signals[sample]['truth_info'][:,0]).astype(int))[ak.argmax(ak.unflatten(preds_sig.reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
            results['stop_samples'][name][sample]['evt']['top_MaxisPair'] = np.array((results['stop_samples'][name][sample]['evt']['top_Maxinputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
            results['stop_samples'][name][sample]['evt']['top_true_inputs'] = np.array(ak.unflatten(data_signals[sample]['x'].detach().cpu().numpy(), np.array(data_signals[sample]['truth_info'][:,0]).astype(int))[ak.argmax(ak.unflatten(data_signals[sample]['y'].reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int)),keepdims=True,axis=-1)][:,0])
            results['stop_samples'][name][sample]['evt']['top_true_isPair'] = np.array((results['stop_samples'][name][sample]['evt']['top_true_inputs'][:,inputs.index('jet2_pT')])==0).reshape(-1).astype(int)
            results['stop_samples'][name][sample]['evt']['top_Matched'] = np.array(ak.max(ak.unflatten(results['stop_samples'][name][sample]['labels'].reshape(-1), np.array(data_signals[sample]['truth_info'][:,0]).astype(int)),axis=-1)).reshape(-1)

    return results

def get_observable(results,model,sample,inputs,reco='top',obs='mass'):
    if obs == 'TopNN_score': return results[sample][model]['evt']['top_Maxscore']
    if obs == 'truth_top_pt': return results[sample][model]['evt']['variables'][:,variables.index('truth_topp_pt')]
    b= vector.array(
        {
            "pt": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('bjet_pT')],
            "phi": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('bjet_phi')],
            "eta": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('bjet_eta')],
            "M": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('bjet_M')],
        }
    )
    j1 = vector.array(
        {
            "pt": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet1_pT')],
            "phi": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet1_phi')],
            "eta": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet1_eta')],
            "M": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet1_M')],
        }
    )
    j2 = vector.array(
        {
            "pt": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet2_pT')],
            "phi": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet2_phi')],
            "eta": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet2_eta')],
            "M": results[sample][model]['evt']['top_Maxinputs'][:,inputs.index('jet2_M')],
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

def get_observable_SPANet(pt,phi,eta,mass,predictions,reco='top',obs='mass'):
    N = len(pt)
    row_values = np.arange(10)
    map = np.tile(row_values, (N, 1))

    pt = pt[np.arange(len(predictions))[:, np.newaxis], predictions]
    phi = phi[np.arange(len(predictions))[:, np.newaxis], predictions]
    eta = eta[np.arange(len(predictions))[:, np.newaxis], predictions]
    mass = mass[np.arange(len(predictions))[:, np.newaxis], predictions]
    for v in [pt,phi,eta,mass]:
        print(v.shape)
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
    else: 
        print('choose reco: top, W')
        return 0
    if obs=='mass': observable = obj.mass
    else: 
        print('choose observable: mass')
        return 0
    return observable

def plot(results,model,sample='bkg',obj='top',obs='mass'):
    if obj=='top': b=np.linspace(0,400,40)
    elif obj=='W': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.set_title(models_name[model])
    WeightEvents = results[sample][model]['evt']['variables'][:,variables.index('WeightEvents')]
    WeightEvents=1
    observable = get_observable(results,model,sample=sample,inputs=inputs,reco=obj,obs=obs)
    ax.hist([     
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    ],
                    bins=b,
                    weights=[
                        WeightEvents*(results[sample][model]['evt']['top_Matched']==0),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']==0),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']==0),
                    ],
                    stacked=True,
                    label=[
                        f'{sample} w/o complete truth top had',
                        f'{sample} w/ top Pair not matched',
                        f'{sample} w/ top Pair matched',
                        f'{sample} w/ top Triplet not matched',
                        f'{sample} w/ top Triplet matched',
                    ],    
                )
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': ax.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': ax.set_xlabel('true top pT [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    out_dir='Plots/Event_level'   
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')  
    fig.savefig(f'{out_dir}/{model}_{sample}_Cand_{obj}_{obs}.png')


def plot_SPANet(model,pt,phi,eta,mass,predictions,sample='stop',obj='top',obs='mass'):
    if obj=='top': b=np.linspace(0,400,40)
    elif obj=='W': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.set_title(models_name[model])
    WeightEvents = results[sample][model]['evt']['variables'][:,variables.index('WeightEvents')]
    WeightEvents=1
    observable = get_observable_SPANet(pt,phi,eta,mass,predictions,reco=obj,obs=obs)
    ax.hist([     
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    ],
                    bins=b,
                    weights=[
                        WeightEvents*(results[sample][model]['evt']['top_Matched']==0),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']==0),
                        WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']==0),
                    ],
                    stacked=True,
                    label=[
                        f'{sample} w/o complete truth top had',
                        f'{sample} w/ top Pair not matched',
                        f'{sample} w/ top Pair matched',
                        f'{sample} w/ top Triplet not matched',
                        f'{sample} w/ top Triplet matched',
                    ],    
                )
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': ax.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': ax.set_xlabel('true top pT [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    out_dir='Plots/Event_level'   
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')  
    fig.savefig(f'{out_dir}/SPANet_{model}_{sample}_Cand_{obj}_{obs}.png')

def plot_multiple_models(results,models,sample='bkg',obj='top',obs='mass'):
    Heigth=int(math.sqrt( len(models) ))+1
    Width=int(math.sqrt(len(models)) )+1
    if obj=='top': b=np.linspace(0,400,40)
    elif obj=='W': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    fig, axs = plt.subplots(Heigth, Width,figsize=(12, 10), dpi=600)
    h = -1
    for i,model in enumerate(models):
        w = (i)%Width
        if w==0: h += 1
        axs[h,w].set_title(models_name[model])
        WeightEvents = results[sample][model]['evt']['variables'][:,variables.index('WeightEvents')]
        WeightEvents=1
        observable = get_observable(results,model,sample=sample,inputs=inputs,reco=obj,obs=obs)
        axs[h,w].hist([     
                        observable,
                        observable,
                        observable,
                        observable,
                        observable,
                        ],
                        bins=b,
                        weights=[
                            WeightEvents*(results[sample][model]['evt']['top_Matched']==0),
                            WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']),
                            WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']),
                            WeightEvents*results[sample][model]['evt']['top_Matched']*(results[sample][model]['evt']['top_Maxscore_label']==0)*(results[sample][model]['evt']['top_true_isPair']==0),
                            WeightEvents*results[sample][model]['evt']['top_Matched']*results[sample][model]['evt']['top_Maxscore_label']*(results[sample][model]['evt']['top_true_isPair']==0),
                        ],
                        stacked=True,
                        label=[
                            f'{sample} w/o complete truth top had',
                            f'{sample} w/ top Pair not matched',
                            f'{sample} w/ top Pair matched',
                            f'{sample} w/ top Triplet not matched',
                            f'{sample} w/ top Triplet matched',
                        ],    
                    )
        axs[h,w].set_ylabel('Events (a.u.)')
        if obj=='top': axs[h,w].set_xlabel(f'top cand {obs} [GeV]',loc='right')
        elif obj=='W': axs[h,w].set_xlabel(f'W cand {obs} [GeV]',loc='right')
        elif obs=='TopNN_score': axs[h,w].set_xlabel('top cand score',loc='right')
        elif obs=='truth_top_pt': axs[h,w].set_xlabel('true top pT [GeV]',loc='right')
        if obs=='TopNN_score': axs[h,w].legend(fontsize=8,loc='upper left')
        else: axs[h,w].legend(fontsize=8,loc='upper right')
    out_dir='Plots/Event_level/Multiple_models'   
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')  
    fig.savefig(f'{out_dir}/{sample}_Cand_{obj}_{obs}.png')

def get_matrix(results,model,metric='auc',seff=0.9):
    M1 = np.arange(500,1601,100)
    M2 = np.arange(0,801,100)
    M2[0]=1
    M2 = np.flip(M2)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
    matrix = np.ones((len(M2),len(M1)))
    for j,m1 in enumerate(M1):
        for i,m2  in enumerate(M2):
            matrix[i,j] = 0 
            MS = (m1 - m2)
            for sample in results['stop_samples'][model].keys():
                if f'_{m1}_{m2}_' in sample:
                    if metric == 'bkg_rej': 
                        matrix[i,j] = 1/results['stop_samples'][model][sample]['fpr'][int(len(results['stop_samples'][model][sample]['fpr'])*seff)]
                    else: matrix[i,j] = results['stop_samples'][model][sample][metric]                
    if metric == 'bkg_rej': 
        matrix[0,0] = 1/results['stop'][model]['fpr'][int(len(results['stop'][model]['fpr'])*seff)]
        matrix[0,1] = 1/results['bkg'][model]['fpr'][int(len(results['bkg'][model]['fpr'])*seff)]
        #matrix[1,0] = 1/results['all'][model]['fpr'][int(len(results['all'][model]['fpr'])*seff)]
    else: 
        matrix[0,0] = results['stop'][model][metric]   
        matrix[0,1] = results['bkg'][model][metric]    
        #matrix[1,0] = results['all'][model][metric]                   
    matrix[matrix == 0] = np.nan         
    ax.matshow(matrix, cmap=plt.cm.Blues, interpolation = 'none', vmin = 0.8)
    xaxis = np.arange(len(M1))
    yaxis = np.arange(len(M2))
    if metric == 'bkg_rej': ax.set_title(f'{metric} at {seff} signal efficiency [ {models_name[model]} ]')
    else: ax.set_title(f'{metric} [ {models_name[model]} ]')
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xlabel('Stop mass')
    ax.set_ylabel('Neutralino mass')
    ax.set_yticklabels(list(np.char.mod('%d', M2)))
    ax.set_xticklabels(list(np.char.mod('%d', M1)))
    ax.xaxis.set_ticks_position('bottom')
    for j,m1 in enumerate(M1):
        for i,m2  in enumerate(M2):
            mess = ''
            color="black"
            fontsize='medium'
            fontstyle='normal'
            if not np.isnan(matrix[i,j]) : 
                if (i == 0) and j == 0: 
                    mess = 'All stop\n'
                    fontsize='large'
                    fontstyle='italic'
                elif (i == 0) and j == 1: 
                    mess = 'All bkg\n'
                    fontsize='large'
                    fontstyle='italic'
                #elif (i == 1) and j == 0: 
                #    mess = 'All\n s+b\n'
                #    fontsize='large'
                #    fontstyle='italic'    
                ax.text(j, i, f'{mess}{matrix[i,j]:.3f}', va='center', ha='center',color=color,fontstyle=fontstyle,fontsize=fontsize)
  
    if metric == 'bkg_rej': 
        out = f'{metric}_{seff}_{model}.png' 
    else: 
        out = f'{metric}_{model}.png'     
    out_dir = f'Plots/{metric}'    
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')    
    fig.savefig(f'{out_dir}/{out}')    
    return matrix    

def get_ratios(matrices,model1,model2,metric='auc',seff=0.9):
    M1 = np.arange(500,1601,100)
    M2 = np.arange(0,801,100)
    M2[0]=1
    M2 = np.flip(M2)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
    if metric == 'bkg_rej': 
        matrix=matrices[model1][f'{metric}_{seff}']/matrices[model2][f'{metric}_{seff}']
        vmin=0.5
        vmax=1.5
    else: 
        matrix=matrices[model1][metric]/matrices[model2][metric]
        vmin=0.95
        vmax=1.05
    ax.matshow(matrix, cmap=plt.cm.RdBu, interpolation = 'none', vmin=vmin, vmax=vmax)
    xaxis = np.arange(len(M1))
    yaxis = np.arange(len(M2))
    if metric == 'bkg_rej': ax.set_title(f'{metric} at {seff} signal efficiency [ {models_name[model1]} / {models_name[model2]} ]')
    else: ax.set_title(f'{metric} - {models_name[model1]} / {models_name[model2]}')
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xlabel('Stop mass')
    ax.set_ylabel('Neutralino mass')
    ax.set_yticklabels(list(np.char.mod('%d', M2)))
    ax.set_xticklabels(list(np.char.mod('%d', M1)))
    ax.xaxis.set_ticks_position('bottom')
    for j,m1 in enumerate(M1):
        for i,m2  in enumerate(M2):
            mess = ''
            color="black"
            fontsize='medium'
            fontstyle='normal'
            if not np.isnan(matrix[i,j]) : 
                if (i == 0) and j == 0: 
                    mess = 'All stop\n'
                    fontsize='large'
                    fontstyle='italic'
                elif (i == 0) and j == 1: 
                    mess = 'All bkg\n'
                    fontsize='large'
                    fontstyle='italic'
                #elif (i == 1) and j == 0: 
                #    mess = 'All\n s+b\n'
                #    fontsize='large'
                #    fontstyle='italic' 
                ax.text(j, i, f'{mess}{matrix[i,j]:.3f}', va='center', ha='center',color=color,fontstyle=fontstyle,fontsize=fontsize)
    if metric == 'bkg_rej': 
        out = f'{metric}_{seff}_ratio_{model1}_{model2}.png' 
    else:  
        out = f'{metric}_ratio_{model1}_{model2}.png'  
    out_dir = f'Plots/{metric}_ratio'        
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')    
    fig.savefig(f'{out_dir}/{out}')         

if __name__ == "__main__":
    
    file = '/raven/u/mvigl/Stop/data/H5_full/Virtual_test.h5'
    with open('/raven/u/mvigl/Stop/TopNN/data/H5/test/filter_sig_FS.txt', "r") as filter:
        samples_sig = [line.strip() for line in filter.readlines()]
    with open('/raven/u/mvigl/Stop/TopNN/data/H5/test/filter_bkg_all.txt', "r") as filter:
        samples_bkg = [line.strip() for line in filter.readlines()]

    idmap='/raven/u/mvigl/Stop/TopNN/data/stop_samples.yaml'
    results = get_results(file,samples_sig,samples_bkg,idmap,models=None)
    dir_out = 'results'
    if (not os.path.exists(dir_out)): os.system(f'mkdir {dir_out}')
    with open(f'{dir_out}/results.h5', 'wb') as f:
        pickle.dump(results, f) 

    print('AUC top_Maxscore_label : ', get_AUC_topMAX(results,'Stop_FS_1000000',sample='stop',label='top_Maxscore_label'))
    print('AUC top_Matched : ', get_AUC_topMAX(results,'Stop_FS_1000000',sample='stop',label='top_Matched'))
    print('accuracy triplets baseline : ',results['stop']['Stop_FS_1000000']['acc_triplets'])
    #with h5py.File('/raven/u/mvigl/Stop/run/pre/H5_spanet_sig_FS/spanet_inputs_test.h5','r') as h5fw :
    #        with h5py.File('/raven/u/mvigl/Stop/TopNN/data/SPANet/evals.h5','r') as evals :
    #            pt = h5fw['INPUTS']['Source']['pt'][:]
    #            eta = h5fw['INPUTS']['Source']['eta'][:]
    #            phi = h5fw['INPUTS']['Source']['phi'][:]
    #            mass = h5fw['INPUTS']['Source']['mass'][:]
    #            predictions = evals['predictions'][0]
#
    print('accuracy baseline : ',results['stop']['Stop_FS_1000000']['acc'])
    #plot_SPANet('Stop_FS_1000000',pt,phi,eta,mass,predictions,sample='stop',obj='top',obs='mass')
    #plot_SPANet('Stop_FS_1000000',pt,phi,eta,mass,predictions,sample='stop',obj='W',obs='mass')


    if True:
        models = [
            'Stop_FS_1000000',
            'Slicing_Full_bkg_1000000',
            'Slicing_Full_200000',
            ]

        for model in models:
            plot(results,model,sample='stop',obj='top',obs='mass')
            plot(results,model,sample='stop',obj='W',obs='mass')
            plot(results,model,sample='bkg',obj='top',obs='mass')
            plot(results,model,sample='bkg',obj='W',obs='mass')
            plot(results,model,sample='stop',obs='TopNN_score')
            plot(results,model,sample='bkg',obs='TopNN_score')
            plot(results,model,sample='stop',obs='truth_top_pt')
            plot(results,model,sample='bkg',obs='truth_top_pt')
        with h5py.File('../SPANet/data/semi_leptonic_ttbar/spanet_inputs_test.h5','r') as h5fw :
            with h5py.File('../SPANet/evals_output/evals.h5','r') as evals :
                pt = h5fw['INPUTS']['Source']['pt'][:]
                eta = h5fw['INPUTS']['Source']['eta'][:]
                phi = h5fw['INPUTS']['Source']['phi'][:]
                mass = h5fw['INPUTS']['Source']['mass'][:]
                predictions = evals['predictions'][0]

        plot_SPANet('Stop_FS_1000000',pt,phi,eta,mass,predictions,sample='stop',obj='top',obs='mass')
        plot_SPANet('Stop_FS_1000000',pt,phi,eta,mass,predictions,sample='stop',obj='W',obs='mass')

        plot_multiple_models(results,models,sample='stop',obj='top',obs='mass')
        plot_multiple_models(results,models,sample='stop',obj='W',obs='mass')
        plot_multiple_models(results,models,sample='bkg',obj='top',obs='mass')
        plot_multiple_models(results,models,sample='bkg',obj='W',obs='mass')
        plot_multiple_models(results,models,sample='stop',obs='TopNN_score')
        plot_multiple_models(results,models,sample='bkg',obs='TopNN_score')
        plot_multiple_models(results,models,sample='stop',obs='truth_top_pt')
        plot_multiple_models(results,models,sample='bkg',obs='truth_top_pt')    

        Heigth=int(math.sqrt( len(models) ))+1
        Width=int(math.sqrt(len(models)) )+1
        b=np.linspace(0,1,100)
        fig, axs = plt.subplots(Heigth, Width,figsize=(12, 10), dpi=600)
        h = -1
        for i,model in enumerate(models):
            w = (i)%Width
            if w==0: h += 1
            #print('w: ', w,' h: ', h )
            axs[h,w].set_title(models_name[model])
            axs[h,w].hist(results['stop'][model]['preds'],bins=b,weights=results['stop']['labels'],histtype='step',density=True,label=f'STOP truth top')
            axs[h,w].hist(results['stop'][model]['preds'],bins=b,weights=1*(results['stop']['labels']==0),histtype='step',density=True,label='STOP no truth top')
            axs[h,w].hist(results['bkg'][model]['preds'],bins=b,weights=results['bkg']['labels'],histtype='step',density=True,label='BKG truth top')
            axs[h,w].hist(results['bkg'][model]['preds'],bins=b,weights=1*(results['bkg']['labels']==0),histtype='step',density=True,label='BKG no truth top')
            axs[h,w].text(0.3, 4.0, (f'AUC STOP = {results["stop"][model]["auc"]:.3f}') )
            axs[h,w].text(0.3, 2.8, (f'AUC BKG   = {results["bkg"][model]["auc"]:.3f}') )
            #axs[h,w].text(0.3, 2.0, (f'AUC ALL    = {results["all"][model]["auc"]:.3f}') )
            axs[h,w].legend(loc='lower center')
            axs[h,w].set_ylabel('Multiplets')
            axs[h,w].set_xlabel('TopNN score',loc='right')
            axs[h,w].semilogy()
        out_dir='Plots/Scores'   
        if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')        
        fig.savefig(f'{out_dir}/Scores.png')

        models = list(results['stop'].keys())[1:]

        MATRICES = {}
        for model in models:
            MATRICES[model] = {}
            MATRICES[model]['auc'] = get_matrix(results,model,metric='auc')
            for seff in [0.6,0.7,0.8,0.9]:
                MATRICES[model][f'bkg_rej_{seff}'] = get_matrix(results,model,metric='bkg_rej',seff=seff)       

        for model in list(models)[1:]:
            get_ratios(MATRICES,model,'Stop_FS_1000000',metric='auc')
            for seff in [0.6,0.7,0.8,0.9]:
                get_ratios(MATRICES,model,'Stop_FS_1000000',metric='bkg_rej',seff=seff)         

    
        models = [  'Stop_FS_10000',
                    'Stop_FS_50000',
                    'Stop_FS_1000000',
                    #'Full_bkg_65000',
                    'Full_bkg_68000',
                    #'Full_bkg_70000',
                    #'Full_bkg_80000',
                    #'Full_bkg_100000',
                    'Full_bkg_200000',
                    'Full_bkg_1000000',
                    'Slicing_Full_bkg_1000000',
                    'Slicing_Full_200000',
              ]

        colors = {
                    'Stop_FS_10000':'darkcyan',
                    'Stop_FS_50000':'blue',
                    'Stop_FS_1000000':'navy',
                    #'Full_bkg_65000':,
                    'Full_bkg_68000':'indianred',
                    #'Full_bkg_70000':,
                    #'Full_bkg_80000':,
                    #'Full_bkg_100000':,
                    'Full_bkg_200000':'orange',
                    'Full_bkg_1000000':'red',
                    'Slicing_Full_bkg_1000000':'maroon',
                    'Slicing_Full_200000':'green',
        }
        fig = plt.figure(figsize=(8, 6), dpi=600)
        ax = fig.add_subplot(4,1,(1,3)) 
        for model in models:
            ax.plot(results['stop'][model]['tpr'],1/results['stop'][model]['fpr'],label=f'{models_name[model]}',color=colors[model]) 
        ax.legend()    
        ax.semilogy()
        ax.set_xlim(0.6,1)
        ax.set_ylim(1,17.5)     
        ax.set_ylabel('Bkg rej')
        plt.setp(ax.get_xticklabels(), visible=False)
        ax = fig.add_subplot(4,1,4)
        for model in models:
            ax.plot(results['stop'][model]['tpr'],results['stop']['Stop_FS_1000000']['fpr']/results['stop'][model]['fpr'],label=f'{models_name[model]}',color=colors[model]) 
        ax.set_xlim(0.6,1)
        ax.set_ylim(0.9,1.05) 
        ax.set_xlabel('Signal efficiency',loc='right')
        out_dir='Plots/Hierarchy'   
        if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')       
        fig.savefig(f'{out_dir}/Bkg_rej.png')