from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
device = get_device()

def idxs_to_var(train,val,branches,var,split):

    train['label'] = (ak.flatten(ak.Array(branches['multiplets'][:split]))[:,-1].to_numpy()+1)/2
    val['label'] = (ak.flatten(ak.Array(branches['multiplets'][split:]))[:,-1].to_numpy()+1)/2

    bj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,0]))[:,:,0]*(ak.Array(branches['bjet1'+var]))
    bj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,1]))[:,:,0]*(ak.Array(branches['bjet2'+var]))
    train['bjet_'+var] = ak.flatten( (bj1 + bj2)[:split] ).to_numpy()
    val['bjet_'+var] = ak.flatten( (bj1 + bj2)[split:] ).to_numpy()

    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,1]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,1]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,1]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,1]*(ak.Array(branches['ljet4'+var]))
    train['jet1_'+var] = ak.flatten( (lj1 + lj2 + lj3 + lj4)[:split] ).to_numpy()
    val['jet1_'+var] = ak.flatten( (lj1 + lj2 + lj3 + lj4)[split:] ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,2]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,2]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,2]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,2]*(ak.Array(branches['ljet4'+var]))
    train['jet2_'+var] = ak.flatten( (lj1 + lj2 + lj3 + lj4)[:split] ).to_numpy()
    val['jet2_'+var] = ak.flatten( (lj1 + lj2 + lj3 + lj4)[split:] ).to_numpy()

    return train, val

def get_data(branches,frac=0.8,vars=['pT','eta','phi','M'],training=True):
    split = int(len(ak.Array(branches['ljetIdxs_saved'][:]))*frac)
    train = {}
    val = {}
    for var in vars:
        train, val = idxs_to_var(train,val,branches,var,split)
        
    train_data = np.hstack( (   train['bjet_pT'][:,np.newaxis],
                                train['jet1_pT'][:,np.newaxis],
                                train['jet2_pT'][:,np.newaxis],
                                train['bjet_eta'][:,np.newaxis],
                                train['jet1_eta'][:,np.newaxis],
                                train['jet2_eta'][:,np.newaxis],
                                train['bjet_phi'][:,np.newaxis],
                                train['jet1_phi'][:,np.newaxis],
                                train['jet2_phi'][:,np.newaxis],
                                train['bjet_M'][:,np.newaxis],
                                train['jet1_M'][:,np.newaxis],
                                train['jet2_M'][:,np.newaxis]) )
                   
    val_data = np.hstack( (     val['bjet_pT'][:,np.newaxis],
                                val['jet1_pT'][:,np.newaxis],
                                val['jet2_pT'][:,np.newaxis],
                                val['bjet_eta'][:,np.newaxis],
                                val['jet1_eta'][:,np.newaxis],
                                val['jet2_eta'][:,np.newaxis],
                                val['bjet_phi'][:,np.newaxis],
                                val['jet1_phi'][:,np.newaxis],
                                val['jet2_phi'][:,np.newaxis],
                                val['bjet_M'][:,np.newaxis],
                                val['jet1_M'][:,np.newaxis],
                                val['jet2_M'][:,np.newaxis]) )
    if training: return train_data, train['label']
    else: return val_data,val['label']

filelist = 'data/train_list.txt'

class CustomDataset(Dataset):
    def __init__(self,filelist,device,training=True):
        self.device = device
        self.x=[]
        self.y=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with uproot.open({fname_mc: "stop1L_NONE;1"}) as tree_mc:
                    branches = tree_mc.arrays()
                    if i ==0:
                        data,target = get_data(branches,training=training)
                    else:
                        data_i,target_1 = get_data(branches,training=training)
                        data = np.concatenate((data,data_i),axis=0)
                        target = np.concatenate((target,target_1),axis=0)
                    i+=1 
        self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.length = len(target)
        print('N data : ',self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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

def train_step(model,data,target,opt,loss_fn):
    model.train()
    preds =  model(data)
    loss = loss_fn(preds,target)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model, loss_fn,train_loader,val_loader,device):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if i==0:
                data, target = train_batch
                data = data.cpu().numpy()
                target = target.cpu().numpy()
            else: 
                data = np.concatenate((data,train_batch[0].cpu().numpy()),axis=0)
                target = np.concatenate((target,train_batch[1].cpu().numpy()),axis=0)
            if (i > 100): break 
        for i, val_batch in enumerate( val_loader ):
            if i==0:
                data_val, target_val = val_batch
                data_val = data_val.cpu().numpy()
                target_val = target_val.cpu().numpy()
            else: 
                data_val = np.concatenate((data_val,val_batch[0].cpu().numpy()),axis=0)
                target_val = np.concatenate((target_val,val_batch[1].cpu().numpy()),axis=0)           

        train_loss = loss_fn(model( torch.from_numpy(data).float().to(device) ).reshape(len(data)),torch.from_numpy(target.reshape(-1)).float().to(device))
        test_loss = loss_fn(model( torch.from_numpy(data_val).float().to(device) ).reshape(len(data_val)),torch.from_numpy(target_val.reshape(-1)).float().to(device))    
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,filelist,device,experiment):
    opt = optim.Adam(model.parameters(), 0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.157]).to(device))
    evals = []
    best_val_loss = float('inf')
    Dataset = CustomDataset(filelist,device,training=True)
    Dataset_val = CustomDataset(filelist,device,training=False)

    best_model_params_path = 'test.pt'
    val_loader = DataLoader(Dataset_val, batch_size=512, shuffle=True)
    for epoch in range (0,100):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(Dataset, batch_size=512, shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data, target = train_batch
            report = train_step(model, data, target, opt, loss_fn )
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader,device) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    
    experiment.end()
    return evals, model    

model = make_mlp(in_features=12,out_features=64,nlayer=4,for_inference=False,binary=True)

print(model)

hyper_params = {
   "learning_rate": 0.001,
   "epochs": 10,
   "batch_size": 512,
}
experiment_name = f'test_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}'
experiment = Experiment(
    api_key = 'r1SBLyPzovxoWBPDLx3TAE02O',
    project_name = 'Stop',
    workspace='mvigl',
    log_graph=True, # Can be True or False.
    auto_metric_logging=True # Can be True or False
    )
Experiment.set_name(experiment,experiment_name)
print(experiment.get_key())
experiment.log_parameter("exp_key", experiment.get_key())
experiment.log_parameters(hyper_params)

E,M = train_loop(model,filelist,device,experiment)