import numpy as np
import uproot
import awkward as ak
import argparse
import pickle
import h5py
import os
import yaml
import ROOT
import onnxruntime as ort

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filelist', help='data',default='/u/mvigl/Stop/TopNN/data/root/list_one.txt')
parser.add_argument('--massgrid', help='massgrid',default='/raven/u/mvigl/Stop/TopNN/data/stop_masses.yaml')
args = parser.parse_args()

batch_size = 200000 

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
          '1600_1','1600_100','1600_200','1600_300','1600_400','1600_500','1600_600','1600_700','1600_800'
          ]
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
                [1600,1],[1600,100],[1600,200],[1600,300],[1600,400],[1600,500],[1600,600],[1600,700],[1600,800]
]

Features = ['multiplets',
            'bjetIdxs_saved',
            'ljetIdxs_saved',
            'bjet1pT',
            'bjet2pT',
            'bjet3pT',
            'ljet1pT',
            'ljet2pT',
            'ljet3pT',
            'ljet4pT',
            'bjet1eta',
            'bjet2eta',
            'bjet3eta',
            'ljet1eta',
            'ljet2eta',
            'ljet3eta',
            'ljet4eta',
            'bjet1phi',
            'bjet2phi',
            'bjet3phi',
            'ljet1phi',
            'ljet2phi',
            'ljet3phi',
            'ljet4phi',
            'bjet1M',
            'bjet2M',
            'bjet3M',
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
            'WeightEvents',
            'WeightEventsbTag',
            'WeightEventselSF',
            'WeightEventsJVT',
            'WeightEventsmuSF',
            'WeightEventsPU',
            'WeightEventsSF_global',
            'WeightEvents_trigger_ele_single',
            'WeightEvents_trigger_mu_single',
            'xsec',
            'WeightLumi',
            'lep1pT',
            'lep1eta',
            'lep1phi',
            'lep1M',
            'MET',
            'METsig',
            'METphi',
            'MET_Soft',
            'MET_Jet',
            'MET_Ele',
            'MET_Muon',
            'mT_METl',
            'dR_bb',
            'dphi_METl',
            'MT2_bb',
            'MT2_b1l1_b2',
            'MT2_b2l1_b1',
            'MT2_min',
            'HT',
            'nbjet',
            'nljet',
            'njet',
            'nlep',
            'nVx',
            'EventNumber',
            ]

def idxs_to_var(branches):

    # dummy filter, here is all events
    filter = (ak.Array(branches['multiplets'])[:,0,-1] > -100)
    rnum_array = (ak.Array(branches['EventNumber'])[:])

    # no good had-top triplet/doublet
    not_matched =  (ak.Array(branches['multiplets'])[filter,0,-1]!=1)
    # no lep-top
    not_matched_l =  (((ak.Array(branches['truth_topp_match'])[filter].to_numpy()==-1)+(ak.Array(branches['truth_topm_match'])[filter].to_numpy()==-1))==0)
    
    length = np.sum(filter)
    vars=['btag','qtag','etag','eta','M','phi','pT']
    inputs = {}
    # fill the 4-momenta info
    # will store as (bjet1,bjet2,ljet1,ljet2,ljet3,ljet4,lep,bjet3,0,0) - important for filling the idxs of reconstruction targets 
    # can add more jets in the future
    for var in vars:
        if var == 'btag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,0] += 1
            inputs[var][:,1] += 1
            inputs[var][:,6] += 1
        elif var == 'qtag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,2] += 1
            inputs[var][:,3] += 1
            inputs[var][:,4] += 1
            inputs[var][:,5] += 1  
        elif var == 'etag':
            inputs[var] = np.zeros((length,10))
            inputs[var][:,7] += 1
        else:
            inputs[var] = np.zeros((length,10))
            inputs[var][:,0] += ak.Array(branches['bjet1'+var][filter]).to_numpy()
            inputs[var][:,1] += ak.Array(branches['bjet2'+var][filter]).to_numpy()
            inputs[var][:,2] += ak.Array(branches['ljet1'+var][filter]).to_numpy()
            inputs[var][:,3] += ak.Array(branches['ljet2'+var][filter]).to_numpy()
            inputs[var][:,4] += ak.Array(branches['ljet3'+var][filter]).to_numpy()
            inputs[var][:,5] += ak.Array(branches['ljet4'+var][filter]).to_numpy()
            inputs[var][:,7] += ak.Array(branches['lep1'+var][filter]).to_numpy()
            inputs[var][:,6] += ak.Array(branches['bjet3'+var][filter]).to_numpy()
            (inputs[var])[inputs[var]==-999]=0.
            (inputs[var])[inputs[var]==-10]=0.
    mask = (inputs['pT']>0)
    inputs['btag'][mask==False]=0.
    inputs['qtag'][mask==False]=0.
    inputs['etag'][mask==False]=0.

    # fill the global features
    met = {
        'MET': branches['MET'][filter].to_numpy(),
        'METsig': branches['METsig'][filter].to_numpy(),
        'METphi': branches['METphi'][filter].to_numpy(),
        'MET_Soft': branches['MET_Soft'][filter].to_numpy(),
        'MET_Jet': branches['MET_Jet'][filter].to_numpy(),
        'MET_Ele': branches['MET_Ele'][filter].to_numpy(),
        'MET_Muon': branches['MET_Muon'][filter].to_numpy(),
        'mT_METl': branches['mT_METl'][filter].to_numpy(),
        'dR_bb': branches['dR_bb'][filter].to_numpy(),
        'dphi_METl': branches['dphi_METl'][filter].to_numpy(),
        'MT2_bb': branches['MT2_bb'][filter].to_numpy(),
        'MT2_b1l1_b2': branches['MT2_b1l1_b2'][filter].to_numpy(),
        'MT2_b2l1_b1': branches['MT2_b2l1_b1'][filter].to_numpy(),
        'MT2_min': branches['MT2_min'][filter].to_numpy(),
        'HT': branches['HT'][filter].to_numpy(),
        'nbjet': branches['nbjet'][filter].to_numpy(),
        'nljet': branches['nljet'][filter].to_numpy(),
        'nVx': branches['nVx'][filter].to_numpy(),
    }

    return mask,inputs,met

def get_data(branches,massgrid,sig=False,number=3456):
    mask,inputs,met = idxs_to_var(branches)
    if sig:
        signal = np.ones(len(mask))
        # signal mass info
        with open(massgrid) as file:
            map = yaml.load(file, Loader=yaml.FullLoader)['samples'] 
        m1=(map[number])[0]
        m2=(map[number])[1]
        M1 = np.ones(len(mask))*m1
        M2 = np.ones(len(mask))*m2
    else:
        signal = np.zeros(len(mask))    
        M1 = -1*np.ones(len(mask))
        M2 = -1*np.ones(len(mask))
    
    Momenta_data = np.array([inputs['M'][:],
                    inputs['pT'][:],
                    inputs['eta'][:],
                    inputs['phi'][:],
                    inputs['btag'][:],
                    inputs['qtag'][:],
                    inputs['etag'][:]]).astype(np.float32).swapaxes(0,1).swapaxes(1,2)
    Momenta_mask = np.array(mask).astype(bool)

    Met_data = np.array([met['MET'][:],
                    met['METsig'][:],
                    met['METphi'][:],
                    met['MET_Soft'][:],
                    met['MET_Jet'][:],
                    met['MET_Ele'][:],
                    met['MET_Muon'][:],
                    met['mT_METl'][:],
                    met['dR_bb'][:],
                    met['dphi_METl'][:],
                    met['MT2_bb'][:],
                    met['MT2_b1l1_b2'][:],
                    met['MT2_b2l1_b1'][:],
                    met['MT2_min'][:],
                    met['HT'][:],
                    met['nbjet'][:],
                    met['nljet'][:],
                    met['nVx'][:],
                    M1,
                    M2,]).astype(np.float32).swapaxes(0,1)[:,np.newaxis,:]
    Met_mask = np.ones((len(Momenta_mask),1)).astype(bool)

    return Momenta_data,Momenta_mask,Met_data,Met_mask
  

def run_in_batches(model_path, Momenta_data,Momenta_mask,Met_data,Met_mask, batch_size, masses,masses_slice):
    #ort_sess = ort.InferenceSession(model_path)
    ort_sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    num_batches = len(Met_data) // batch_size
    if len(Met_data) % batch_size != 0:
        num_batches += 1
    print('num batches : ',num_batches)
    outputs = {}
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
            (batch_inputs['Met_data'][:,:,-2:])=masses_slice[j]
            batch_outputs = ort_sess.run(None, {'Momenta_data': batch_inputs['Momenta_data'],
                              'Momenta_mask': batch_inputs['Momenta_mask'],
                              'Met_data': batch_inputs['Met_data'],
                              'Met_mask': batch_inputs['Met_mask']})
        
            if i == 0: outputs[mass] = batch_outputs[4]
            else: outputs[mass]=np.concatenate((outputs[mass],batch_outputs[4]),axis=0)
            #else: 
            #    for k in range(len(outputs)):
            #        outputs[mass][k]=np.concatenate((outputs[mass][k],batch_outputs[k]),axis=0)
    #with open(f'evals_param.pkl', 'wb') as pickle_file:
    #    pickle.dump(outputs, pickle_file)    
    return outputs

def save_single(args):
        with open(args.filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                sig=False
                if '_Signal_' in filename: sig=True
                with uproot.open({filename: "stop1L_NONE;1"}) as tree:
                    number = number = filename[(filename.index("TeV.")+4):(filename.index(".stop1L"))]
                    branches = tree.arrays(Features)
                    Momenta_data,Momenta_mask,Met_data,Met_mask = get_data(branches,args.massgrid,sig=sig,number=number)
                    scores = run_in_batches("/raven/u/mvigl/TopReco/SPANet/spanet_param_weights_log_norm.onnx", Momenta_data,Momenta_mask,Met_data,Met_mask,batch_size,masses,masses_slice)

                file = ROOT.TFile(f'{filename}', "UPDATE")
                tree = file.Get("stop1L_NONE;1")  
                for mass in masses:
                    buffer = np.zeros(1, dtype=np.float32)
                    new_branch = tree.Branch(f'SB_{mass}', buffer, f'SB_{mass}/F')
                    for i in range(tree.GetEntries()):
                        tree.GetEntry(i)
                        buffer[0] = scores[mass][i][1]  # Example data, replace with actual computation
                        new_branch.Fill()
                
                tree.Write("", ROOT.TObject.kOverwrite)
                file.Close()

                        
if __name__ == '__main__':
    save_single(args)
