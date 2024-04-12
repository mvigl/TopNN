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
import torch.onnx
import torch.nn as nn
from onnx2keras import onnx_to_keras
import json
import keras


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

if __name__ == "__main__":
    
    weights = '/u/mvigl/Stop/run/Final/Stop_FS_scaler_nodes128_layers4_lr0.0001_bs512_1000000.pt'
    weights = '../public/TopNN/models/Stop_FS_scaler_nodes128_layers4_lr0.0001_bs512_1000000.pt'
    in_features = 12
    out_features = int(weights[weights.index("nodes")+5:weights.index("_layers")])
    nlayer = int(weights[weights.index("layers")+6:weights.index("_lr")])
    model = make_mlp(in_features=in_features,out_features=out_features,nlayer=nlayer,for_inference=True,binary=True)
    model = load_weights(model,weights,device)

    class PyTorchModel(torch.nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.fc1 = torch.nn.Linear(12, 128)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(128, 128)
            self.fc3 = torch.nn.Linear(128, 128)
            self.fc4 = torch.nn.Linear(128, 128)
            self.fc5 = torch.nn.Linear(128, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.relu(x)
            x = self.fc5(x)
            x = self.sigmoid(x)
            return x
    my_model = PyTorchModel()
    print(my_model)

    with torch.no_grad():
        my_model.fc1.weight.copy_(model[0].weight)
        my_model.fc1.bias.copy_(model[0].bias)
        my_model.fc2.weight.copy_(model[2].weight)
        my_model.fc2.bias.copy_(model[2].bias)
        my_model.fc3.weight.copy_(model[4].weight)
        my_model.fc3.bias.copy_(model[4].bias)
        my_model.fc4.weight.copy_(model[6].weight)
        my_model.fc4.bias.copy_(model[6].bias)
        my_model.fc5.weight.copy_(model[8].weight)
        my_model.fc5.bias.copy_(model[8].bias)

# Ensure that the weights have been copied successfully
    for param1, param2 in zip(model.parameters(), my_model.parameters()):
        assert torch.allclose(param1, param2)


    import onnx
    import numpy as np
    import torch
    from torch.autograd import Variable
    import tensorflow as tf

    from pt2keras import Pt2Keras
    from pt2keras import converter

    import torch.nn.functional as F
    import torch.nn as nn

    import onnx
    from onnx2keras import onnx_to_keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    from keras.models import Model  # , Sequential, load_model
    from keras.layers import Dense, Input  # Activation, LSTM, Masking, Dropout

    inputs = Input(shape=(12,))
    x = Dense(128, activation='relu')(inputs)
    for i in range(3):
        x = Dense(128, activation='relu')(x)
    pred = Dense(1, activation='sigmoid')(x)
    keras_model = Model(inputs=inputs, outputs=pred)

    keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(keras_model.summary())

    # Define a function to convert PyTorch weights to NumPy arrays
    def torch_to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Copy weights from PyTorch to Keras
    keras_model.layers[1].set_weights([torch_to_numpy(my_model.fc1.weight.T), torch_to_numpy(my_model.fc1.bias)])
    keras_model.layers[2].set_weights([torch_to_numpy(my_model.fc2.weight.T), torch_to_numpy(my_model.fc2.bias)])
    keras_model.layers[3].set_weights([torch_to_numpy(my_model.fc3.weight.T), torch_to_numpy(my_model.fc3.bias)])
    keras_model.layers[4].set_weights([torch_to_numpy(my_model.fc4.weight.T), torch_to_numpy(my_model.fc4.bias)])
    keras_model.layers[5].set_weights([torch_to_numpy(my_model.fc5.weight.T), torch_to_numpy(my_model.fc5.bias)])

    with torch.no_grad():
        my_model.eval()
        tensor_x = torch.rand((10, 12), dtype=torch.float32)
        print(tensor_x)
        pred1 = my_model(tensor_x)
        pred2 = keras_model.predict(tensor_x)

    keras_model.save_weights("models/Baseline_TopNN.weights.h5")    
    print(pred1)   
    print(pred2) 



    def get_variables_json():
        """
        Make a file that specifies the input variables and
        transformations as JSON, so that the network can be used with lwtnn
        """

        # This is a more 'traditional' network with one set of inputs so
        # we just have to name the variables in one input node. In more
        # advanced cases we could have multiple input nodes, some of which
        # might operate on sequences.
        toptag_variables = [
            {
                # Note this is not the same name we use in the file! We'll
                # have to make the log1p transformation in the C++ code to
                # build this variable.
                'name': 'b_Pt', "scale": 1, "offset": 0
            },
            {'name': 'j1_Pt', "scale": 1, "offset": 0},
            {'name': 'j2_Pt', "scale": 1, "offset": 0},
            {'name': 'b_eta', "scale": 1, "offset": 0},
            {'name': 'j1_eta', "scale": 1, "offset": 0},
            {'name': 'j2_eta', "scale": 1, "offset": 0},
            {'name': 'b_phi', "scale": 1, "offset": 0},
            {'name': 'j1_phi', "scale": 1, "offset": 0},
            {'name': 'j2_phi', "scale": 1, "offset": 0},
            {'name': 'b_M', "scale": 1, "offset": 0},
            {'name': 'j1_M', "scale": 1, "offset": 0},
            {'name': 'j2_M', "scale": 1, "offset": 0},
                            ]

        # note that this is a list of output nodes, where each node can
        # have multiple output values. In principal we could also add a
        # regression output to the same network (i.e. the b-hadron pt) but
        # that's a more advanced subject.
        outputs = [
            {
                'name': 'classes',
                'labels': ['top']
             }
                   ]

        # lwtnn expects a specific format that allows multiple input and
        # output nodes, so we have to dress the above information a bit.
        final_dict = {
            'input_sequences': [],
            'inputs': [
                {
                    'name': 'toptag_variables',
                    'variables': toptag_variables
                 }
                       ],
            'outputs': outputs
                      }

        return final_dict

    with open(f'models/architecture_0_baseline.json', 'w') as arch_file:
        arch_file.write(keras_model.to_json(indent=2))
    with open(f'models/architecture_1_baseline.json', 'w') as arch_file:
        arch_file.write(keras_model.to_json(indent=2))

    # also write out the variable specification
    with open(f'models/variables_baseline.json', 'w') as vars_file:
        json.dump(get_variables_json(), vars_file)

    print(keras.__version__)