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
import random
import h5py

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--nodes', type=int,  help='epochs',default='64')
parser.add_argument('--nlayers', type=int,  help='epochs',default='4')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Stop/data/H5_full/Virtual_full.h5')
parser.add_argument('--filterlist', help='filterlist',default='/raven/u/mvigl/Stop/TopNN/data/H5/filter_sig_all.txt')
parser.add_argument('--scaler',  action='store_true', help='use scaler', default=False)
parser.add_argument('--project_name', help='project_name',default='Stop_final')
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')
parser.add_argument('--mess', help='message',default='Full')

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

def get_idxmap(filelist,dataset='train'):
    idxmap = {}
    offset = 0 
    with open(filelist) as f:
        for line in f:
            filename = line.strip()
            data_index = (filename.index("/mc"))
            name = (filename[data_index+1:])
            with h5py.File(filename, 'r') as Data:
                if dataset == 'train':
                    length = int(len(Data['labels'])*0.9)
                else:
                    length = int(len(Data['labels'])*0.05)
                idxmap[name] = [int(offset), int(offset+length)]
                offset += length
    return idxmap

def range_to_string(index, idxmap):
    for name, range in idxmap.items():
        if range[0] <= index < range[1]:
            return name, range[0], range[1]-range[0]
    return "No matching range found"  


    
class CustomDataset_maps(Dataset):
    def __init__(self, idxmap,file,dataset='train'):
        self.idxmap = idxmap
        self.dataset = dataset
        self.file = file
        self.length = list(self.idxmap.values())[-1][-1]
        print("N data : ", self.length)
        
    def __getitem__(self, index):
        name, offset, length = range_to_string(index,self.idxmap)
        x = []
        with h5py.File(self.file, 'r') as f:
            if self.dataset == 'val': 
                index += int(length*0.9)
            x = f[name]['multiplets'][index-offset]
            y = f[name]['labels'][index-offset]
        return torch.tensor(x).float(),torch.tensor(y).float()
    
    def __len__(self):
        return self.length        

class CustomDataset(Dataset):
    def __init__(self,file,name,dataset='train'):
        self.dataset = dataset
        self.file = file
        self.x=[]
        self.y=[]
        with h5py.File(self.file, 'r') as f:
            length = len(f[name]['labels'])
            idx_train = int(length*0.9)
            idx_val = int(length*0.95)
            if self.dataset == 'train':
                data = f[name]['multiplets'][:idx_train]
                target = f[name]['labels'][:idx_train]
            else:
                data = f[name]['multiplets'][idx_train:idx_val]
                target = f[name]['labels'][idx_train:idx_val]  
        
        self.x = torch.from_numpy(data).float().to(device)    
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.length = len(target)
        #print(self.dataset, " sample , ", "N data : ", self.length)
        
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
    loss = loss_fn(preds.reshape(-1),target.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model,loss_fn,file,samples):
    print('validation...')
    for name in samples:
        Dataset_val = CustomDataset(file,name,dataset='val')
        Dataset_train = CustomDataset(file,name,dataset='train')
        val_loader = DataLoader(Dataset_val, batch_size=hyper_params["batch_size"], shuffle=True)
        train_loader = DataLoader(Dataset_train, batch_size=hyper_params["batch_size"], shuffle=True)
        if len(Dataset_train) < 1: continue
        if len(Dataset_val) < 1: continue
        with torch.no_grad():
            model.eval()
            for i, train_batch in enumerate( train_loader ):
                if i==0:
                    data,target = train_batch
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
                if (i > 100): break 

    train_loss = loss_fn(model(data).reshape(-1),target.reshape(-1))
    test_loss = loss_fn(model(data_val).reshape(-1),target_val.reshape(-1))    
    print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
    return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,file,samples,device,experiment,hyper_params,path):
    opt = optim.Adam(model.parameters(), hyper_params["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.]).to(device)) #16.4 in stop
    evals = []
    best_val_loss = float('inf')
    best_model_params_path = path

    for epoch in range (0,hyper_params["epochs"]):
        print(f'epoch: {epoch+1}') 
        random.shuffle(samples)
        for name in samples:
            Dataset_train = CustomDataset(file,name,dataset='train')
            if len(Dataset_train) < 1: continue
            train_loader = DataLoader(Dataset_train, batch_size=hyper_params["batch_size"], shuffle=True)
            for i, train_batch in enumerate( train_loader ):
                data, target = train_batch
                report = train_step(model, data, target, opt, loss_fn )
        evals.append(eval_fn(model, loss_fn,file,samples) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    
    experiment.end()
    return evals, model    

hyper_params = {
    "filter": args.filterlist,
    "data": args.data,
    "scaler": '',
    "nodes": args.nodes,
    "nlayer": args.nlayers,
    "learning_rate": args.lr,
    "epochs": args.ep,
    "batch_size": args.bs,
}

experiment_name = f'{args.mess}_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}'
path = f'nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pt'
if args.scaler:
    experiment_name = f'{args.mess}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}'
    path = f'{args.mess}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pt'
    hyper_params["scaler"] = f'{args.mess}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}.pkl'

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
print(device)
with h5py.File(args.data, 'r') as f:
    samples = list(f.keys())    

if args.filterlist != '': 
    with open(args.filterlist, "r") as file:
        samples = [line.strip() for line in file.readlines()]

E,M = train_loop(model,args.data,samples,device,experiment,hyper_params,path)

log_model(experiment, model, model_name = experiment_name )