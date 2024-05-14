import h5py
import numpy as np
import vector
import os
import matplotlib.pyplot as plt
import vector 
from sklearn.metrics import roc_curve,auc
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', help='data',default='data/root/list_sig_FS_testing.txt')
parser.add_argument('--evals', help='evals',default='data/root/list_sig_FS_testing.txt')
args = parser.parse_args()


def get_observable(pt,phi,eta,mass,predictions,detection_probabilities,thr=0.,reco='top',obs='mass'):
    pt = pt[np.arange(len(predictions))[:, np.newaxis], predictions]
    phi = phi[np.arange(len(predictions))[:, np.newaxis], predictions]
    eta = eta[np.arange(len(predictions))[:, np.newaxis], predictions]
    mass = mass[np.arange(len(predictions))[:, np.newaxis], predictions]
    for v in [pt,phi,eta,mass]:
        v[detection_probabilities<thr,2] = 0
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

def get_observable_matched(pt,phi,eta,mass,targets,reco='top',obs='mass'):
    pt = pt[np.arange(len(targets))[:, np.newaxis], targets]
    phi = phi[np.arange(len(targets))[:, np.newaxis], targets]
    eta = eta[np.arange(len(targets))[:, np.newaxis], targets]
    mass = mass[np.arange(len(targets))[:, np.newaxis], targets]
    for v in [pt,phi,eta,mass]:
        v[targets==-1] = 0.
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

with h5py.File(args.data,'r') as h5fw :    
    with h5py.File(args.evals,'r') as evals :
        print(h5fw['CLASSIFICATIONS']['EVENT'].keys())
        print(h5fw['INPUTS'].keys())
        met = h5fw['INPUTS']['Met']['MET'][:]
        weights = np.ones(len(met))
        for weight in [ 'WeightEvents',
                        'WeightEventsbTag',
                        'WeightEventselSF',
                        'WeightEventsJVT',
                        'WeightEventsmuSF',
                        'WeightEventsPU',
                        'WeightEventsSF_global',
                        'WeightEvents_trigger_ele_single',
                        'WeightEvents_trigger_mu_single',
                        'xsec',
                        'WeightLumi'
                        ]:
            weights*=h5fw['truth_info'][weight][:]
        pt = h5fw['INPUTS']['Momenta']['pt'][:]
        eta = h5fw['INPUTS']['Momenta']['eta'][:]
        phi = h5fw['INPUTS']['Momenta']['phi'][:]
        mass = h5fw['INPUTS']['Momenta']['mass'][:]
        bees = np.sum(pt[:,[0,1,6]]>0,axis=-1)
        print(h5fw['TARGETS']['ht']['b'][:])
        print(h5fw['TARGETS']['lt']['b'][:])
        signal = evals['signal'][:]
        labels = h5fw['CLASSIFICATIONS']['EVENT']['signal'][:]
        match_pred = evals['match'][:]
        match_label = h5fw['CLASSIFICATIONS']['EVENT']['match'][:]
        predictions = evals['predictions_ht'][0]
        predictions_probabilities = np.nan_to_num(evals['assignment_probabilities'][0])
        detection_probabilities = evals['detection_probabilities'][0]
        masks = h5fw['INPUTS']['Momenta']['MASK'][:]
        targets = evals['targets_ht'][0]
        masks_in = h5fw['INPUTS']['Momenta']['MASK'][:]
        print(h5fw['truth_info'].keys())
        masses = np.array([h5fw['truth_info']['M1'][:],h5fw['truth_info']['M2'][:]])
        is_matched = h5fw['truth_info']['is_matched'][:]
        truth_top_pt = h5fw['truth_info']['truth_topp_pt'][:]
        truth_top_min_dR_m = h5fw['truth_info']['truth_top_min_dR_m'][:]/1000
        truth_topp_pt = h5fw['truth_info']['truth_topp_pt'][:]
        is_pair = np.sum((targets >= 0),axis=-1)==2 
        truth_Wp_pt = h5fw['truth_info']['truth_Wp_pt'][:]
        truth_top_min_dR_m_jj = h5fw['truth_info']['truth_top_min_dR_m_jj'][:]/1000
        truth_topp_match = h5fw['truth_info']['truth_topp_match'][:]
        truth_topm_match = h5fw['truth_info']['truth_topm_match'][:]
        match_category = np.maximum(truth_topp_match,truth_topm_match)+2

match_max = np.argmax(match_pred,axis=-1)
masses_one = np.unique(masses[0])[1:]
masses_two = np.unique(masses[1])[1:]

matching= {
    7: 'hadronic top w/ matched b-jet and 2 l-jets',
    6: 'hadronic top w/ matched b-jet and 1 l-jets',
    5: 'hadronic top w/ matched b-jet',
    4: 'hadronic top w/ no matched b-jet and 2 l-jets',
    3: 'hadronic top w/ no matched b-jet and 1 l-jet',
    2: 'hadronic top w/ no matched jets',
    1:'leptonic top',
    0:'no top',
}

def plot_categories(match=match_category,sample='stop',obj='top',obs='mass',algo='SPANet',thr=0.2):
    if obj=='top': b=np.linspace(0,400,40)
    elif obj=='W': b=np.linspace(0,140,40)
    elif obj=='top_pair': b=np.linspace(0,400,40)
    elif obj=='W_pair': b=np.linspace(0,140,40)
    if obs=='TopNN_score': b=np.linspace(0,1,40)
    elif obs=='truth_top_pt': b=np.linspace(0,1000,40)
    elif obs=='truth_top_min_dR_m': b=np.linspace(0,400,40)
    elif obs=='pt': b=np.linspace(0,1000,40)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plt.title(f'{algo}')
    stacked=True
    #ax.set_title(models_name[model])
    #WeightEvents = results[sample][model]['evt']['variables'][:,variables.index('WeightEvents')]
    WeightEvents=1
    
    if obs == 'pt':
        if obj=='top':
            observable_truth = truth_topp_pt
        else:
            observable_truth = truth_Wp_pt    
    elif obs == 'mass':
        if obj=='top':
            observable_truth = truth_top_min_dR_m
        else:
            observable_truth = truth_top_min_dR_m_jj               
    if algo == 'matching':                 
        observable = get_observable_matched(pt,phi,eta,mass,targets,reco=obj,obs=obs)
    elif algo == 'SPANet':  
        if obs=='detection_probability': 
            observable = detection_probabilities
            b=np.linspace(0,1,100)
        elif obs=='prediction_probability': 
            observable = predictions_probabilities
            b=np.linspace(0.,1,40)    
        elif obs=='truth_top_pt': observable = truth_top_pt
        elif obs=='truth_top_min_dR_m': observable = truth_top_min_dR_m
        else: observable = get_observable(pt,phi,eta,mass,predictions,detection_probabilities,thr=thr,reco=obj,obs=obs) 
    elif algo == 'truth': observable = observable_truth
    observable[observable<0]=0.        
    if algo == 'matching': observable[truth_topp_match<4]=-100.
    
    ax.hist([     
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    ],
                    bins=b,
                    weights=[
                        1*(match==0),
                        1*(match==1),
                        1*(match==2),
                        1*(match==3),
                        1*(match==4),
                        1*(match==5),
                        1*(match==6),
                        1*(match==7),
                    ],
                    stacked=stacked,
                    label=[
                        f'{matching[0]}',
                        f'{matching[1]}',
                        f'{matching[2]}',
                        f'{matching[3]}',
                        f'{matching[4]}',
                        f'{matching[5]}',
                        f'{matching[6]}',
                        f'{matching[7]}',
                    ],    
                )
    if obs not in ['mass','detection_probability','prediction_probability']:
        observable_truth[observable_truth<0]=0.   
        ax.hist([     
                        observable_truth,
                        ],
                        bins=b,
                        weights=[
                            np.ones_like(observable_truth),
                        ],
                        stacked=stacked,
                        color='black',
                        histtype='step',
                        linestyle='dashed',
                        label=[
                            f'Truth',
                        ],  
                        lw=2
                    )
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': ax.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': ax.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': ax.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=8,loc='upper right')
    if (obs=='detection_probability') or (obs=='prediction_probability'): 
        ax.semilogy()
    #ax.semilogy()    
    if ((algo == 'SPANet') and (obs!='detection_probability')): fig.savefig(f'Categories/Cand_{obj}_{obs}_{algo}_thr_{thr}.png')
    else: fig.savefig(f'Categories/Cand_{obj}_{obs}_{algo}.png')
#for obj in ['top','W']:
#    for obs in ['mass','pt']:
#        for thr in [0,0.1,0.2,0.3,0.5]:
#            plot_categories(obj=obj,obs=obs,algo='SPANet',thr=thr)  
#        plot_categories(obj=obj,obs=obs,algo='baseline')  
#        plot_categories(obj=obj,obs=obs,algo='truth')  
#        plot_categories(obj=obj,obs=obs,algo='matching')  

def get_auc(labels,signal,weights,masses,m1=1000,m2=100,region='all'):
    if np.sum(labels*(masses[0]==m1)*(masses[1]==m2))==0: return
    if region == 'all':
        w_bkg = 1*weights*(labels==0)
        w_sig = weights*labels*(masses[0]==m1)*(masses[1]==m2)
        filter = ((masses[0]==m1)*(masses[1]==m2)+(labels==0)).astype(bool)
        fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels[filter],signal[filter,1],drop_intermediate=True)
        Auc_sig = auc(fpr_sig,tpr_sig)
    elif region == '1b':    
        w_bkg = 1*weights*(labels==0)*(met>230)*(bees==1)
        w_sig = weights*labels*(masses[0]==m1)*(masses[1]==m2)*(met>230)*(bees==1)
        filter = (((masses[0]==m1)*(masses[1]==m2)+(labels==0))*(met>230)*(bees==1)).astype(bool)
        fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels[filter],signal[filter,1],drop_intermediate=True)
        Auc_sig = auc(fpr_sig,tpr_sig)
    elif region == '2b':    
        w_bkg = 1*weights*(labels==0)*(met>230)*(bees>1)
        w_sig = weights*labels*(masses[0]==m1)*(masses[1]==m2)*(met>230)*(bees>1) 
        filter = (((masses[0]==m1)*(masses[1]==m2)+(labels==0))*(met>230)*(bees>1)).astype(bool)
        fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels[filter],signal[filter,1],drop_intermediate=True)
        Auc_sig = auc(fpr_sig,tpr_sig)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    #WeightEvents
    ax.hist(signal[:,1],bins=np.linspace(0,1,40),weights=w_bkg,density=False,label='bkg',histtype='step')
    ax.hist(signal[:,1],bins=np.linspace(0,1,40),weights=w_sig,density=False,label='sig',histtype='step')
    ax.legend()
    ax.set_title(f'm1: {m1} m2: {m2} AUC : {Auc_sig}')
    #ax.set_ylim(0.00001,10)
    ax.semilogy()
    out_dir = f'signals'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    out_dir = f'signals/{region}'
    if (not os.path.exists(out_dir)): os.system(f'mkdir {out_dir}')
    fig.savefig(f'signals/{region}/SvsB_{m1}_{m2}.png')

def plot_single_categories(match=match_category,match_max=match_max,sample='sig',obj='top',obs='mass',algo='SPANet',thr=0.2,category=5,
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
    plt.title(f'{algo} {sample}')
    stacked=True
    #ax.set_title(models_name[model])
    #WeightEvents = results[sample][model]['evt']['variables'][:,variables.index('WeightEvents')]
    
    if obs == 'pt':
        if obj=='top':
            observable_truth = truth_topp_pt
        else:
            observable_truth = truth_Wp_pt    
    elif obs == 'mass':
        if obj=='top':
            observable_truth = truth_top_min_dR_m
        else:
            observable_truth = truth_top_min_dR_m_jj             
    if algo == 'matching':                 
        observable = get_observable_matched(pt,phi,eta,mass,targets,reco=obj,obs=obs)
    elif algo == 'SPANet':  
        if obs=='detection_probability': 
            observable = detection_probabilities
            b=np.linspace(0,1,100)
        elif obs=='prediction_probability': 
            observable = predictions_probabilities
            b=np.linspace(0.9,1,40)    
        elif obs=='truth_top_pt': observable = truth_top_pt
        elif obs=='truth_top_min_dR_m': observable = truth_top_min_dR_m
        else: observable = get_observable(pt,phi,eta,mass,predictions,detection_probabilities,thr=thr,reco=obj,obs=obs) 
    elif algo == 'truth': observable = observable_truth
    observable_match = get_observable_matched(pt,phi,eta,mass,targets,reco=obj,obs=obs)
    observable_truth[observable_truth<0]=0.   
    observable_match[observable_match<0]=0.   
    observable[observable<0]=0.        
    label = np.ones_like(observable)
    if sample == 'sig': label = (labels==1)
    elif sample == 'bkg': label = (labels==0)
    if algo == 'matching': observable[truth_topp_match<4]=-100.
    ax.hist([     
                    observable_truth,
                    ],
                    bins=b,
                    weights=[
                        1*weights*(match==category)*(label),
                    ],
                    stacked=stacked,
                    histtype='step',
                    color='black',
                    label=[
                        f'TRUTH {matching[category]}',
                    ],    
                    lw=2
                )
    if category > 5:
        ax.hist([     
                    observable_match,
                    ],
                    bins=b,
                    weights=[
                        1*weights*(match==category)*(label),
                    ],
                    stacked=stacked,
                    histtype='step',
                    color=colors[category],
                    label=[
                        f'TRUTH MATCHED {matching[category]}',
                    ],    
                    lw=2
                )
    ax.hist([     
                    observable,
                    ],
                    bins=b,
                    weights=[
                        1*weights*(match==category)*(label),
                    ],
                    stacked=stacked,
                    linestyle='dashed',
                    histtype='step',
                    color=colors[category],
                    label=[
                        f'SPANet TRUTH MATCHED {matching[category]}',
                    ],    
                    lw=2
                )
    if algo == 'SPANet':  
        ax.hist([     
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    observable,
                    ],
                    bins=b,
                    alpha=0.8,
                    weights=[
                        1*weights*(match_max==category)*(match==0)*(label),
                        1*weights*(match_max==category)*(match==1)*(label),
                        1*weights*(match_max==category)*(match==2)*(label),
                        1*weights*(match_max==category)*(match==3)*(label),
                        1*weights*(match_max==category)*(match==4)*(label),
                        1*weights*(match_max==category)*(match==5)*(label),
                        1*weights*(match_max==category)*(match==6)*(label),
                        1*weights*(match_max==category)*(match==7)*(label),
                    ],
                    stacked=stacked,
                    color=colors[:8],
                    label=[
                        f'{matching[0]}',
                        f'{matching[1]}',
                        f'{matching[2]}',
                        f'{matching[3]}',
                        f'{matching[4]}',
                        f'{matching[5]}',
                        f'{matching[6]}',
                        f'{matching[7]}',
                    ],    
                )
    ax.set_ylabel('Events (a.u.)')
    if obj=='top': ax.set_xlabel(f'top cand {obs} [GeV]',loc='right')
    elif obj=='W': ax.set_xlabel(f'W cand {obs} [GeV]',loc='right')
    elif obs=='TopNN_score': ax.set_xlabel('top cand score',loc='right')
    elif obs=='truth_top_pt': ax.set_xlabel('true top pT [GeV]',loc='right')
    elif obs=='truth_top_min_dR_m': ax.set_xlabel('true top Mass [GeV]',loc='right')
    if obs=='TopNN_score': ax.legend(fontsize=8,loc='upper left')
    else: ax.legend(fontsize=4,loc='upper right')
    if (obs=='detection_probability') or (obs=='prediction_probability'): 
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


def get_match_plot(match_pred,sample='sig',density=True):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    for i in range(8):
        w=1*(match_label==i)
        if sample == 'sig': w*=(labels==1)
        elif sample == 'bkg': w*=(labels==0)
        ax.hist(np.argmax(match_pred,axis=-1),bins=np.linspace(-0.5,7.5,9),weights=w,density=density,histtype='step',label=f'{matching[i]}')
    #ax.semilogy()
    if density: ax.set_ylim(0,1)
    ax.legend(fontsize=2,loc='upper left')    
    if density: fig.savefig(f'match_{sample}_norm.png')
    else: fig.savefig(f'match_{sample}.png')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

if __name__ == "__main__":
    
    plot_categories(obj='top',obs='prediction_probability',algo='SPANet')
    plot_categories(obj='top',obs='detection_probability',algo='SPANet')
    get_match_plot(match_pred,sample='sig',density=False)
    get_match_plot(match_pred,sample='bkg',density=False)
    get_match_plot(match_pred,sample='sig',density=True)
    get_match_plot(match_pred,sample='bkg',density=True)

    fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels,signal[:,1],drop_intermediate=True)
    Auc_sig = auc(fpr_sig,tpr_sig)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=1*weights*(labels==0),density=True,label='bkg',histtype='step')
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=weights*labels,density=True,label='sig',histtype='step')
    ax.legend()
    ax.set_title(f'all - AUC : {Auc_sig}')
    ax.set_ylim(0.01,100)
    ax.semilogy()
    fig.savefig(f'SvsB.png')

    fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels[(met>230)*(bees>1)],signal[(met>230)*(bees>1),1])
    Auc_sig = auc(fpr_sig,tpr_sig)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=1*weights*(labels==0)*(met>230)*(bees>1),density=True,label='bkg',histtype='step')
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=labels*weights*(met>230)*(bees>1),density=True,label='sig',histtype='step')
    ax.legend()
    ax.set_title(f'2b - AUC : {Auc_sig}')
    ax.set_ylim(0.01,100)
    ax.semilogy()
    fig.savefig(f'SvsB_met230_2b.png')

    fpr_sig, tpr_sig, thresholds_sig = roc_curve(labels[(met>230)*(bees==1)],signal[(met>230)*(bees==1),1])
    Auc_sig = auc(fpr_sig,tpr_sig)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=1*weights*(labels==0)*(met>230)*(bees==1),density=True,label='bkg',histtype='step')
    ax.hist(signal[:,1],bins=np.linspace(0,1,100),weights=weights*labels*(met>230)*(bees==1),density=True,label='sig',histtype='step')
    ax.legend()
    ax.set_title(f'1b - AUC : {Auc_sig}')
    ax.set_ylim(0.01,100)
    ax.semilogy()
    fig.savefig(f'SvsB_met230_1b.png')

    for sample in ['sig','bkg']:
        for category in [0,1,2,3,4,5,6,7]:
            for obj in ['top','W']:
                for obs in ['mass','pt']:
                    if (obj=='W' and obs=='pt'): continue
                    plot_single_categories(sample=sample,obj=obj,obs=obs,algo='SPANet',thr=0,category=category,colors=colors)  

    #stop            
    for m1 in masses_one:
        for m2 in masses_two:
            for region in ['all','1b','2b']:
                print(region,m1,m2)
                get_auc(labels,signal,weights,masses,m1=m1,m2=m2,region=region)     
    #dm
    for region in ['all','1b','2b']:
        get_auc(labels,signal,weights,masses,m1=0,m2=0,region=region)                       