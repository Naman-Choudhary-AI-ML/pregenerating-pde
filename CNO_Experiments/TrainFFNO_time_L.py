from pytorch_lightning import Trainer
from CNO_timeModule_CIN import CNO_time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from FNO import FNO_time
from FFNO import FFNO_time


import copy
import json
import os
import sys

import pandas as pd
import torch

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.00005,      #0.00075, 
        "weight_decay": 1e-6,
        "lr_scheduler": "cosine", # or "step"
        "scheduler_step": None,               #1,
        "scheduler_gamma": None, #0.9,
        "epochs": 400,
        "batch_size": 16,         
        "time_steps": 19,          # How many time steps to select?
        "dt": 1,                  # What is the time step? (1 means include entire traj, 2 means taking every other step, etc.
        "training_samples": 400,   # How many training samples?
        "mixing": False,  # Set True to enable mixing experiment
        "alpha": 0.25,   # Percentage of hole data (e.g., 0.15 = 15%)
        "hole_data_path": "/data/user_data/vhsingh/dataset/scaled_hole_location.npy",
        "nohole_data_path": "/data/user_data/vhsingh/dataset/scaled_NS_Regular.npy",
        "time_input": 1,          # Should we include time in the input channels?
        "allowed": 'one',         # All2ALL (train) - all , or One2All (train) - one2all, AR training - one
        "cluster": True,          # Something internal (don't bother)
    }
    
    model_architecture_ = {
        "modes_x": 16,
        "modes_y": 16,
        "width": 64,
        "factor": 4,
        "n_ff_layers": 2,
        "share_weight": True,
        "ff_weight_norm": True,
        "layer_norm": False,
        "retrain_fno": 42,
        "N_layers": 4,
        }
    
    # AVAILABLE EXPERIMENTS:
    # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
    # "ns_pwc_t:
    # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
    # "rich_mesh", "rayl_tayl" "kolmogorov"
    # "wave_seismic", "wave_gauss", "allen_cahn"
    # "airfoil", "poisson_gauss", "helmholtz"
    
    # FOR PRETRAINING CNO-FM: which_example = "eul_ns_mix1"
    
    # WHAT IS THE EXPERIMENT?
    which_example = "ns_custom"
    
    folder = f"/data/user_data/vhsingh/baseline_experiments/LDC_Cylinder/hole_location/{training_properties['allowed']}"
    os.makedirs(folder, exist_ok=True)


    
else:
    raise ValueError("To many args")
    
    

#---------------------------------------------------------    
cluster = True  # We always tun on cluster

# model_architecture_["batch_norm"] = True if model_architecture_["batch_norm"] in [1, True] else False
# training_properties["time_input"] = True if training_properties["time_input"] in [1, True] else False
# model_architecture_["is_att"]     = True if model_architecture_["is_att"]     in [1, True] else False
# training_properties["cluster"]    = True if training_properties["cluster"]    in [1, True] else False

# if model_architecture_["nl_dim"] in ["023"]:
#     model_architecture_["nl_dim"] = [0,2,3]
# elif model_architecture_["nl_dim"] in ["123"]:
#     model_architecture_["nl_dim"] = [1,2,3]
# elif model_architecture_["nl_dim"] in ["23"]:
#     model_architecture_["nl_dim"] = [2,3]

#---------------------------------------------------------
if not cluster: # Set the defaulf folder
    folder = "--- PROVIDE THE FOLDER TO SAVE THE MODEL (no cluster) ----"  

if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

#---------------------------------------------------------
# Load parameters related to the specific experiment -- "DataLoaders/all_experiments.json"

Dict_EXP = json.load( open( "DataLoaders/all_experiments.json" ))
if which_example in Dict_EXP:
    loader_dict =  Dict_EXP[which_example]
else:
    raise ValueError("Please specify different benchmark")

# loader_dict: INFORMATION ABOUT THE EXPERIMENT, TRAINING, etc -- VERY IMPORTANT!
loader_dict["which"] = which_example
loader_dict["time_input"] = training_properties["time_input"]
loader_dict["cluster"] = training_properties["cluster"]
loader_dict["num_samples"] = training_properties["training_samples"]
loader_dict["dt"] = training_properties["dt"]
loader_dict["time_steps"] = training_properties["time_steps"]
loader_dict["lr_scheduler"] = training_properties.get("lr_scheduler", "step")
loader_dict["max_epochs"] = training_properties["epochs"]
loader_dict["alpha"] = training_properties["alpha"]
loader_dict["mixing"] = training_properties.get("mixing", False)
loader_dict["hole_path"] = training_properties.get("hole_data_path", None)
loader_dict["nohole_path"] = training_properties.get("nohole_data_path", None)



#---------------------------------------------------------
# Which transitions during the training are allowed?
_allowed = []
if "include_zero" in loader_dict and loader_dict["include_zero"]:
    start_t = 0
else:
    start_t = 1
if training_properties['allowed'] == 'all':
    for t in range(start_t,loader_dict["time_steps"]+1):
        _allowed.append(t)
elif training_properties['allowed']  == "one2all":
    _allowed = None
elif training_properties['allowed'] == 'one':
    _allowed = [1]
loader_dict["allowed_tran"] = _allowed

#---------------------------------------------------------
# Initialize CNO
model   = FFNO_time(in_dim =  loader_dict["in_dim"],
                    out_dim = loader_dict["out_dim"],               
                    modes_x= model_architecture_["modes_x"],
                    modes_y=model_architecture_["modes_y"],
                    width = model_architecture_["width"],
                    n_layers = model_architecture_["N_layers"],
                    factor = model_architecture_["factor"],
                    n_ff_layers = model_architecture_["n_ff_layers"],
                    share_weight = model_architecture_["share_weight"],
                    ff_weight_norm = model_architecture_["ff_weight_norm"],
                    layer_norm = model_architecture_["layer_norm"],                                                   

                    lr = training_properties["learning_rate"],
                    weight_decay = training_properties["weight_decay"],

                    loader_dictionary = loader_dict,)

#---------------------------------------------------------

ver = 0 # Just a random string to be added to the model name

checkpoint_callback = ModelCheckpoint(dirpath = folder+"/model"+str(ver), monitor='mean_val_l')
early_part = 3
early_stop_callback = EarlyStopping(monitor="mean_val_l", patience=100)

lr_monitor = LearningRateMonitor(logging_interval='epoch')  # or 'step' if you prefer finer resolution
# logger = TensorBoardLogger(save_dir=folder, version=ver, name="logs")
logger = WandbLogger(
    project="GeoFNO1",
    name=f"FFNO_hole_location_Autoregressive",
    save_dir=folder,
    config={**training_properties, **model_architecture_}  # logs hyperparams too
)

trainer = Trainer(devices = 1,
                max_epochs = training_properties["epochs"],
                callbacks = [checkpoint_callback,early_stop_callback, lr_monitor],
                logger=logger)
trainer.fit(model)
trainer.validate(model)
trainer.test(model)

#---------------------------------------------------------