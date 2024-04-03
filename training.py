from comet_ml import Experiment,ExistingExperiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from sklearn.preprocessing import StandardScaler
import h5py
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', type=float,  help='learning rate',default='0.0001')
parser.add_argument('--bs', type=int,  help='batch size',default='512')
parser.add_argument('--ep', type=int,  help='epochs',default='50')
parser.add_argument('--nodes', type=int,  help='nodes',default='64')
parser.add_argument('--nlayers', type=int,  help='layers',default='4')
parser.add_argument('--maxsamples', type=int,  help='maxsamples',default='1')
parser.add_argument('--data', help='data',default='/raven/u/mvigl/Stop/data/H5_full/Virtual_train.h5')
parser.add_argument('--filterlist', help='filterlist',default='/raven/u/mvigl/Stop/TopNN/data/H5/filter_all.txt')#/raven/u/mvigl/Stop/TopNN/data/H5/filter_sig_FS.txt
parser.add_argument('--scaler',  action='store_true', help='use scaler', default=False)
parser.add_argument('--slicing',  action='store_true', help='slicing', default=False)
parser.add_argument('--project_name', help='project_name',default='Stop_final')
parser.add_argument('--api_key', help='api_key',default='r1SBLyPzovxoWBPDLx3TAE02O')
parser.add_argument('--ws', help='workspace',default='mvigl')
parser.add_argument('--mess', help='message',default='Full')
parser.add_argument('--null',  action='store_true', help='null', default=False)

args = parser.parse_args()

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device
device = get_device()
   

class CustomDataset(Dataset):
    def __init__(self,file,samples,dataset='train',maxsamples=1,epoch=0,scaler_path=''):
        self.dataset = dataset
        self.file = file
        self.x=[]
        self.y=[]
        num_stop=0
        num_bkg=0
        i=0
        with h5py.File(self.file, 'r') as f:
            for name in samples:
                length = len(f[name]['labels'])
                #print(name,' Ndata: ',length)
                length_train = int(length*0.95)
                idx_train = length_train
                lower_bound = 0
                if ((maxsamples!=1) and (length_train > maxsamples)): 
                    quotient, remainder = divmod(length_train, maxsamples)
                    if remainder > 2000: slices = np.concatenate( ((np.ones(quotient)*maxsamples), np.array([remainder])), axis=0).astype(int)
                    else: slices = (np.ones(quotient)*maxsamples).astype(int)
                    id = (epoch - (len(slices)))%len(slices)
                    lower_bound = np.sum(slices[:(id)])
                    idx_train = (lower_bound+slices[id])
                if i==0:
                        data = f[name]['multiplets'][lower_bound:idx_train]
                        target = f[name]['labels'][lower_bound:idx_train]
                else:
                    data = np.concatenate((data,f[name]['multiplets'][lower_bound:idx_train]),axis=0)
                    target = np.concatenate((target,f[name]['labels'][lower_bound:idx_train]),axis=0)        
                if '_Signal_' in name: num_stop += idx_train-lower_bound
                else: num_bkg += idx_train-lower_bound
                i+=1    


        self.scaler = StandardScaler()
        if scaler_path !='' : 
            if (self.dataset == 'train'): 
                X_norm = self.scaler.fit_transform(data)
                with open(scaler_path,'wb') as f:
                    pickle.dump(self.scaler, f)
            else:         
                with open(scaler_path,'rb') as f:
                    self.scaler = pickle.load(f)
                X_norm = self.scaler.transform(data)
            self.x = torch.from_numpy(X_norm).float().to(device)
        else: self.x = torch.from_numpy(data).float().to(device)    
        
        self.y = torch.from_numpy(target.reshape(-1,1)).float().to(device)
        self.length = len(target)
        self.w = np.ones(self.length)
        if (num_bkg != 0) and (num_stop!=0): 
            weight = num_bkg/num_stop
            self.w[:num_stop] *= weight  
        self.w = torch.from_numpy(self.w).float().to(device)    
        print(self.dataset, " Data : ", self.length)
        
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]

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

def train_step(model,data,target,opt,w):
    model.train()
    preds =  model(data)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.6]).to(device),weight=w) #16.4 in stop
    loss = loss_fn(preds.reshape(-1),target.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()
    return {'loss': float(loss)}

def eval_fn(model,file,samples):
    print('validation...')
    num_stop_train=0
    num_bkg_train=0
    num_stop_val=0
    num_bkg_val=0
    i=0
    maxsamples = 200000
    with h5py.File(file, 'r') as f:
        for name in samples:
            length = len(f[name]['labels'])
            length_train = int(length*0.95)
            length_val = int(length-1)
            if length_train < 1: continue
            if length_val < 1: continue
            idx_train = length_train
            idx_val = length_val
            if length_train > maxsamples: idx_train = maxsamples 
            if (length_val-length_train) > maxsamples: idx_val = length_train+maxsamples 

            if i==0:
                data = f[name]['multiplets'][:idx_train]
                target = f[name]['labels'][:idx_train]
                data_val = f[name]['multiplets'][length_train:idx_val]
                target_val = f[name]['labels'][length_train:idx_val]  
            else: 
                data = np.concatenate((data,f[name]['multiplets'][:idx_train]),axis=0)
                target = np.concatenate((target,f[name]['labels'][:idx_train]),axis=0)
                data_val = np.concatenate((data_val,f[name]['multiplets'][length_train:idx_val]),axis=0)
                target_val = np.concatenate((target_val,f[name]['labels'][length_train:idx_val]),axis=0)
            if '_Signal_' in name: 
                num_stop_train += idx_train
                num_stop_val += idx_val-length_train
            else: 
                num_bkg_train += idx_train
                num_bkg_val += idx_val-length_train
            i+=1        
        
    data = torch.from_numpy(data).float().to(device)    
    target = torch.from_numpy(target.reshape(-1,1)).float().to(device)
    data_val = torch.from_numpy(data_val).float().to(device)    
    target_val = torch.from_numpy(target_val.reshape(-1,1)).float().to(device)

    w_train = np.ones(len(target))
    w_val = np.ones(len(target_val))
    if (num_bkg_train != 0) and (num_stop_train!=0): 
        weight_train = num_bkg_train/num_stop_train
        w_train[:num_stop_train] *= weight_train  
    if (num_bkg_val != 0) and (num_stop_val!=0): 
        weight_val = num_bkg_val/num_stop_val
        w_val[:num_stop_val] *= weight_val      
    w_train = torch.from_numpy(w_train).float().to(device)
    w_val = torch.from_numpy(w_val).float().to(device)
    
    loss_fn_train = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.6]).to(device),weight=w_train)
    loss_fn_val = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.6]).to(device),weight=w_val)
    with torch.no_grad():
        model.eval()
        train_loss = loss_fn_train(model(data).reshape(-1),target.reshape(-1))
        test_loss = loss_fn_val(model(data_val).reshape(-1),target_val.reshape(-1))    
        print(f'train_loss: {float(train_loss)} | test_loss: {float(test_loss)}')
        return {'test_loss': float(test_loss), 'train_loss': float(train_loss)}
    

def train_loop(model,file,samples,device,experiment,hyper_params,path):
    opt = optim.Adam(model.parameters(), hyper_params["learning_rate"])
    evals = []
    best_val_loss = float('inf')
    best_model_params_path = path
    if not hyper_params["slicing"]: Dataset_train = CustomDataset(file,samples,dataset='train',maxsamples=hyper_params["maxsamples"],scaler_path=hyper_params["scaler"])
    for epoch in range (0,hyper_params["epochs"]):
        print(f'epoch: {epoch+1}') 
        if hyper_params["slicing"]: Dataset_train = CustomDataset(file,samples,dataset='train',maxsamples=hyper_params["maxsamples"],epoch=epoch)
        train_loader = DataLoader(Dataset_train, batch_size=hyper_params["batch_size"], shuffle=True)
        for i, train_batch in enumerate( train_loader ):
            data, target, w = train_batch
            report = train_step(model, data, target, opt, w )
        evals.append(eval_fn(model,file,samples) )         
        val_loss = evals[epoch]['test_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)
        experiment.log_metrics({"train_loss": evals[epoch]['train_loss'], "val_loss": val_loss}, epoch=(epoch))
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states    
    experiment.end()
    return evals, model    

 

hyper_params = {
    "slicing": args.slicing,
    "maxsamples": args.maxsamples,
    "message": args.mess,
    "filter": args.filterlist,
    "data": args.data,
    "scaler": '',
    "nodes": args.nodes,
    "nlayer": args.nlayers,
    "learning_rate": args.lr,
    "epochs": args.ep,
    "batch_size": args.bs,
}

if hyper_params["slicing"]: hyper_params["message"] = "Slicing_"+hyper_params["message"]
experiment_name = f'{hyper_params["message"]}_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{args.maxsamples}'
path = f'{hyper_params["message"]}_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{args.maxsamples}.pt'
if args.scaler:
    experiment_name = f'{hyper_params["message"]}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{args.maxsamples}'
    path = f'{hyper_params["message"]}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{args.maxsamples}.pt'
    hyper_params["scaler"] = f'{hyper_params["message"]}_scaler_nodes{hyper_params["nodes"]}_layers{hyper_params["nlayer"]}_lr{hyper_params["learning_rate"]}_bs{hyper_params["batch_size"]}_{args.maxsamples}.pkl'

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

with open(args.filterlist, "r") as file:
    samples = [line.strip() for line in file.readlines()]

if __name__ == "__main__":

    E,M = train_loop(model,args.data,samples,device,experiment,hyper_params,path)
    log_model(experiment, model, model_name = experiment_name )