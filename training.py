from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import uproot
import awkward as ak
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--nodes', type=int,  help='epochs',default='64')
parser.add_argument('--nlayers', type=int,  help='epochs',default='4')
parser.add_argument('--data', help='data',default='../../TopNN/train_list.txt')
parser.add_argument('--scaler',  action='store_true', help='use scaler', default=True)
parser.add_argument('--project_name', help='project_name',default='Stop_final')
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')

args = parser.parse_args()

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
device = get_device()

def split_data(length,array,dataset='train'):
    idx_train = int(length*0.9)
    idx_val = int(length*0.95)
    if dataset=='train': return array[:idx_train]
    if dataset=='val': return array[idx_train:idx_val]    
    if dataset=='test': return array[idx_val:]    
    else:       
        print('choose: train, val, test')
        return 0       

def idxs_to_var(out,branches,var,dataset):
    length = len(ak.Array(branches['ljetIdxs_saved'][:]))

    out['label'] = (ak.flatten(  split_data(length,ak.Array(branches['multiplets']),dataset=dataset)  )[:,-1].to_numpy()+1)/2
    bj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,0]))[:,:,0]*(ak.Array(branches['bjet1'+var]))
    bj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['bjetIdxs_saved'][:,1]))[:,:,0]*(ak.Array(branches['bjet2'+var]))
    out['bjet_'+var] = ak.flatten(   split_data(length,(bj1 + bj2),dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,1]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,1]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,1]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,1]*(ak.Array(branches['ljet4'+var]))
    out['jet1_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),dataset=dataset)   ).to_numpy()
    lj1 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,0]))[:,:,2]*(ak.Array(branches['ljet1'+var]))
    lj2 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,1]))[:,:,2]*(ak.Array(branches['ljet2'+var]))
    lj3 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,2]))[:,:,2]*(ak.Array(branches['ljet3'+var]))
    lj4 = (ak.Array(branches['multiplets'][:])==ak.Array(branches['ljetIdxs_saved'][:,3]))[:,:,2]*(ak.Array(branches['ljet4'+var]))
    out['jet2_'+var] = ak.flatten(   split_data(length,(lj1 + lj2 + lj3 + lj4),dataset=dataset)   ).to_numpy()

    return out

def get_data(branches,vars=['pT','eta','phi','M'],dataset='train'):
    output = {}
    for var in vars:
        output = idxs_to_var(output,branches,var,dataset)
        
    out_data = np.hstack(   (   output['bjet_pT'][:,np.newaxis],
                                output['jet1_pT'][:,np.newaxis],
                                output['jet2_pT'][:,np.newaxis],
                                output['bjet_eta'][:,np.newaxis],
                                output['jet1_eta'][:,np.newaxis],
                                output['jet2_eta'][:,np.newaxis],
                                output['bjet_phi'][:,np.newaxis],
                                output['jet1_phi'][:,np.newaxis],
                                output['jet2_phi'][:,np.newaxis],
                                output['bjet_M'][:,np.newaxis],
                                output['jet1_M'][:,np.newaxis],
                                output['jet2_M'][:,np.newaxis]
                            ) 
                        )
                   
    return out_data,output['label']


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
            ]

class CustomDataset(Dataset):
    def __init__(self,filelist,device,Features,scaler_path='',dataset='train'):
        self.device = device
        self.x=[]
        self.y=[]
        i=0
        with open(filelist) as f:
            for line in f:
                filename = line.strip()
                print('reading : ',filename)
                with uproot.open({filename: "stop1L_NONE;1"}) as tree:
                    branches = tree.arrays(Features)
                    if i ==0:
                        data,target = get_data(branches,dataset=dataset)
                    else:
                        data_i,target_1 = get_data(branches,dataset=dataset)
                        data = np.concatenate((data,data_i),axis=0)
                        target = np.concatenate((target,target_1),axis=0)
                    i+=1 
        if scaler_path != '':
            self.scaler = StandardScaler()      
            if dataset=='train': 
                data = self.scaler.fit_transform(data)
                with open(scaler_path,'wb') as f:
                    pickle.dump(self.scaler, f)
            else: 
                with open(scaler_path,'rb') as f:
                    self.scaler = pickle.load(f)
                data = self.scaler.transform(data)
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

def eval_fn(model, loss_fn,train_loader,val_loader):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ): 
            if i==0:
                data, target = train_batch
            else: 
                data = torch.cat((data,train_batch[0]),axis=0)
                target = torch.cat((target,train_batch[1]),axis=0)
            if (i > 100): break 
        for i, val_batch in enumerate( val_loader ):
            if i==0:
                data_val, target_val = val_batch
            else: 
                data_val = torch.cat((data_val,val_batch[0]),axis=0)
                target_val = torch.cat((target_val,val_batch[1]),axis=0)           

        train_loss = loss_fn(model(data).reshape(len(data)),target.reshape(-1))
        test_loss = loss_fn(model(data_val).reshape(len(data_val)),target_val.reshape(-1))    
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,filelist,device,experiment,Features,hyper_params,path):
    opt = optim.Adam(model.parameters(), hyper_params["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.157]).to(device))
    evals = []
    best_val_loss = float('inf')
    best_model_params_path = path
    Dataset = CustomDataset(filelist,device,Features,hyper_params["scaler"],dataset='train')
    Dataset_val = CustomDataset(filelist,device,Features,hyper_params["scaler"],dataset='val')

    val_loader = DataLoader(Dataset_val, batch_size=hyper_params["batch_size"], shuffle=True)
    for epoch in range (0,hyper_params["epochs"]):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(Dataset, batch_size=hyper_params["batch_size"], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data, target = train_batch
            report = train_step(model, data, target, opt, loss_fn )
        evals.append(eval_fn(model, loss_fn,train_loader,val_loader) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    
    experiment.end()
    return evals, model    

hyper_params = {
    "scaler": '',
    "nodes": args.nodes,
    "nlayer": args.nlayers,
    "learning_rate": args.lr,
    "epochs": args.ep,
    "batch_size": args.bs,
}

experiment_name = f'nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}'
path = f'nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pt'
if args.scaler:
    experiment_name = f'Scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}'
    path = f'Scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pt'
    hyper_params["scaler"] = f'Scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pkl'

experiment = Experiment(
    api_key = args.api_key,
    project_name = args.project_name,
    workspace = args.ws,
    log_graph=True, # Can be True or False.
    auto_metric_logging=True # Can be True or False
    )
Experiment.set_name(experiment,experiment_name)
print(experiment.get_key())
experiment.log_parameter("exp_key", experiment.get_key())
experiment.log_parameters(hyper_params)

model = make_mlp(in_features=12,out_features=hyper_params["nodes"],nlayer=hyper_params["nlayer"],for_inference=False,binary=True)
print(model)
model.to(device)

E,M = train_loop(model,args.data,device,experiment,Features,hyper_params,path)

log_model(experiment, model, model_name = experiment_name )