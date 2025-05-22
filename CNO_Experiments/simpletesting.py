import os
import json
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from CNO_timeModule_CIN import CNO_time

def load_configurations(train_config_path, net_config_path):
    """
    Reads the training properties and network architecture files (CSV format with keys in the index)
    and returns two dictionaries.
    """
    train_df = pd.read_csv(train_config_path, header=None, index_col=0)
    net_df = pd.read_csv(net_config_path, header=None, index_col=0)
    
    training_properties = train_df[1].to_dict()
    net_architecture = net_df[1].to_dict()
    
    return training_properties, net_architecture

def build_loader_dict(exp_json_path, which_example, training_properties):
    """
    Loads the experiment configuration from a JSON file and updates it with key values from
    training properties.
    """
    with open(exp_json_path, "r") as f:
        Dict_EXP = json.load(f)
    
    if which_example in Dict_EXP:
        loader_dict = Dict_EXP[which_example]
    else:
        raise ValueError("Experiment {} not found in JSON.".format(which_example))
    
    # Update the loader dictionary with training details
    loader_dict["which"] = which_example
    loader_dict["time_input"] = True if training_properties["time_input"] in [1, "1", True, "True"] else False
    loader_dict["cluster"] = True if training_properties["cluster"] in [1, "1", True, "True"] else False
    loader_dict["num_samples"] = int(training_properties["training_samples"])
    loader_dict["dt"] = int(training_properties["dt"])
    loader_dict["time_steps"] = int(training_properties["time_steps"])
    loader_dict["lr_scheduler"] = training_properties.get("lr_scheduler", "step")
    loader_dict["max_epochs"] = int(training_properties["epochs"])
    loader_dict["alpha"] = float(training_properties["alpha"])
    loader_dict["mixing"] = True if training_properties.get("mixing", "False") in [1, "1", True, "True"] else False
    loader_dict["hole_path"] = training_properties.get("hole_data_path", None)
    loader_dict["nohole_path"] = training_properties.get("nohole_data_path", None)
    
    # Determine allowed transitions based on the "allowed" flag in training properties.
    _allowed = []
    if "include_zero" in loader_dict and loader_dict["include_zero"]:
        start_t = 0
    else:
        start_t = 1
    if training_properties["allowed"] == "all":
        for t in range(start_t, loader_dict["time_steps"] + 1):
            _allowed.append(t)
    elif training_properties["allowed"] == "one2all":
        _allowed = None
    elif training_properties["allowed"] == "one":
        _allowed = [1]
    loader_dict["allowed_tran"] = _allowed

    return loader_dict

def main():
    # ---------------------------------------------------
    # 1. Specify File Paths & Experiment Name
    # ---------------------------------------------------
    # Set the file paths (update these with your actual paths)
    train_config_path = "/data/user_data/namancho/CNO_experiments/FPO_Cylinder_824/Mixing/one/0.0/training_properties.txt"
    net_config_path   = "/data/user_data/namancho/CNO_experiments/FPO_Cylinder_824/Mixing/one/0.0/net_architecture.txt"
    exp_json_path     = "./DataLoaders/all_experiments.json"  # Update this path accordingly
    checkpoint_path   = "/data/user_data/namancho/CNO_experiments/FPO_Cylinder_824/Mixing/one/0.0/model0/epoch=340-step=333839.ckpt"  # Update with your ckpt path
    
    # The experiment name as used during training
    which_example = "ns_custom"
    
    # ---------------------------------------------------
    # 2. Load Saved Configurations
    # ---------------------------------------------------
    training_properties, net_architecture = load_configurations(train_config_path, net_config_path)
    
    # ---------------------------------------------------
    # 3. Cast Values to the Correct Types
    # ---------------------------------------------------
    # For training properties
    training_properties["learning_rate"] = float(training_properties["learning_rate"])
    training_properties["weight_decay"] = float(training_properties["weight_decay"])
    training_properties["epochs"] = int(training_properties["epochs"])
    training_properties["batch_size"] = int(training_properties["batch_size"])
    training_properties["time_steps"] = int(training_properties["time_steps"])
    training_properties["dt"] = int(training_properties["dt"])
    training_properties["training_samples"] = int(training_properties["training_samples"])
    training_properties["alpha"] = float(training_properties["alpha"])
    
    # For network architecture
    net_architecture["N_layers"] = int(net_architecture["N_layers"])
    net_architecture["channel_multiplier"] = int(net_architecture["channel_multiplier"])
    net_architecture["N_res"] = int(net_architecture["N_res"])
    net_architecture["N_res_neck"] = int(net_architecture["N_res_neck"])
    # Convert nl_dim to list if stored as string.
    if net_architecture["nl_dim"] in ["023"]:
        net_architecture["nl_dim"] = [0, 2, 3]
    elif net_architecture["nl_dim"] in ["123"]:
        net_architecture["nl_dim"] = [1, 2, 3]
    elif net_architecture["nl_dim"] in ["23"]:
        net_architecture["nl_dim"] = [2, 3]
    
    # ---------------------------------------------------
    # 4. Build the Loader Dictionary by Merging Experiment JSON and Training Properties
    # ---------------------------------------------------
    loader_dict = build_loader_dict(exp_json_path, which_example, training_properties)
    # Ensure certain keys are properly typed (if not already present, they must be provided by the JSON)
    loader_dict["in_dim"] = int(loader_dict["in_dim"])
    loader_dict["out_dim"] = int(loader_dict["out_dim"])
    loader_dict["time_steps"] = int(loader_dict["time_steps"])
    
    # ---------------------------------------------------
    # 5. Initialize the Model Using Loader Dictionary and Network Architecture
    # ---------------------------------------------------
    # Here, we assume the in_size is fixed (e.g. 128) or taken from your experiment.
    model = CNO_time(
        in_dim = loader_dict["in_dim"],
        in_size = 128,  # Update if in_size is stored differently
        N_layers = net_architecture["N_layers"],
        N_res = net_architecture["N_res"],
        N_res_neck = net_architecture["N_res_neck"],
        channel_multiplier = net_architecture["channel_multiplier"],
        batch_norm = True if net_architecture["batch_norm"] in [1, "1", True, "True"] else False,
        out_dim = loader_dict["out_dim"],
        activation = net_architecture["activation"],
        time_steps = loader_dict["time_steps"],
        is_time = int(net_architecture["is_time"]),
        nl_dim = net_architecture["nl_dim"],
        lr = float(training_properties["learning_rate"]),
        batch_size = int(training_properties["batch_size"]),
        weight_decay = float(training_properties["weight_decay"]),
        scheduler_step = training_properties["scheduler_step"],
        scheduler_gamma = training_properties["scheduler_gamma"],
        loader_dictionary = loader_dict,
        is_att = True if net_architecture["is_att"] in [1, "1", True, "True"] else False,
        patch_size = int(net_architecture["patch_size"]),
        dim_multiplier = float(net_architecture["dim_multiplier"]),
        depth = int(net_architecture["depth"]),
        heads = int(net_architecture["heads"]),
        dim_head_multiplier = float(net_architecture["dim_head_multiplier"]),
        mlp_dim_multiplier = float(net_architecture["mlp_dim_multiplier"]),
        emb_dropout = float(net_architecture["emb_dropout"])
    )
    
    # ---------------------------------------------------
    # 6. Load the Checkpoint and Move the Model to the Proper Device
    # ---------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    # ---------------------------------------------------
    # 7. Setup the WandbLogger to Resume the Existing Run
    # ---------------------------------------------------
    wandb_run_id = "pgsqwk0t"  # Replace with your previous wandb run ID
    wandb_logger = WandbLogger(
        project="GeoFNO1",  # Same project used during training
        id=wandb_run_id,
        resume="must",
        save_dir=os.path.dirname(train_config_path)
    )
    
    # ---------------------------------------------------
    # 8. Create a Trainer and Run Testing
    # ---------------------------------------------------
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if device == "cuda" else "cpu",
        logger=wandb_logger,
    )
    
    trainer.test(model)
    
    # Optionally finalize the wandb run.
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
