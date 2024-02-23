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
import pickle
import h5py

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--nodes', type=int,  help='epochs',default='64')
parser.add_argument('--nlayers', type=int,  help='epochs',default='4')
parser.add_argument('--data', help='data',default='../../TopNN/data/H5/list_sig_AF3.txt')
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
            print('idxmap: ',filename)
            with h5py.File(filename, 'r') as Data:
                length = len(Data['labels'][:])
                if dataset=='train': length = int(length*0.9)
                if dataset=='val': length = int(length*0.05)
                idxmap[filename] = np.arange(offset,offset+int(length),dtype=int)
                offset += int(length)
    return idxmap

def create_integer_file_map(idxmap):
    integer_file_map = {}
    file_names = list(idxmap.keys())
    file_vectors = list(idxmap.values())
    for i, file in enumerate(file_names):
        print('integer_file_map: ',file)
        vector = file_vectors[i]
        for integer in vector:
            if integer in integer_file_map:
                integer_file_map[integer].append(file)
            else:
                integer_file_map[integer] = [file]

    return integer_file_map

class CustomDataset(Dataset):
    def __init__(self, idxmap,integer_file_map,dataset='train'):
        self.integer_file_map = integer_file_map
        self.length = len(integer_file_map)
        self.idxmap = idxmap
        self.dataset = dataset
        print("N data : ", self.length)
        
    def __getitem__(self, index):
        file_path = self.integer_file_map[index][0]
        offset = np.min(self.idxmap[file_path])
        x = []
        with h5py.File(file_path, 'r') as f:
            if self.dataset == 'val': 
                index += int(len(f['labels'][:])*0.9)
            x = f['multiplets'][index-offset]
            y = f['labels'][index-offset]
        return torch.tensor(x).float(),torch.tensor(y).float()
    
    def __len__(self):
        return self.length    


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

def eval_fn(model, loss_fn,train_loader,val_loader,device):
    with torch.no_grad():
        model.eval()
        for i, train_batch in enumerate( train_loader ):
            if i==0:
                data = train_batch[0]
                target = train_batch[1]
                data.to(device)
                target.to(device)
            else: 
                data = torch.cat((data,train_batch[0].to(device) ),axis=0)
                target = torch.cat((target,train_batch[1].to(device) ),axis=0)
            if (i > 100): break 
        for i, val_batch in enumerate( val_loader ):
            if i==0:
                data_val = val_batch[0]
                target_val = val_batch[1]
                data_val.to(device)
                target_val.to(device)
            else: 
                data_val = torch.cat((data_val,val_batch[0].to(device) ),axis=0)
                target_val = torch.cat((target_val,val_batch[1].to(device) ),axis=0)           

        train_loss = loss_fn(model(data).reshape(-1),target.reshape(-1))
        test_loss = loss_fn(model(data_val).reshape(-1),target_val.reshape(-1))    
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,filelist,device,experiment,hyper_params,path):
    opt = optim.Adam(model.parameters(), hyper_params["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.]).to(device)) #16.4 in stop
    evals = []
    best_val_loss = float('inf')
    best_model_params_path = path

    idxmap = get_idxmap(filelist,dataset='train')
    idxmap_val = get_idxmap(filelist,dataset='val')
    integer_file_map = create_integer_file_map(idxmap)
    integer_file_map_val = create_integer_file_map(idxmap_val)
    Dataset = CustomDataset(idxmap,integer_file_map,dataset='train')
    Dataset_val = CustomDataset(idxmap_val,integer_file_map_val,dataset='val')

    val_loader = DataLoader(Dataset_val, batch_size=hyper_params["batch_size"], shuffle=True)
    for epoch in range (0,hyper_params["epochs"]):
        print(f'epoch: {epoch+1}') 
        train_loader = DataLoader(Dataset, batch_size=hyper_params["batch_size"], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data = train_batch[0].to(device)
            target = train_batch[1].to(device)
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

hyper_params = {
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

E,M = train_loop(model,args.data,device,experiment,hyper_params,path)

log_model(experiment, model, model_name = experiment_name )