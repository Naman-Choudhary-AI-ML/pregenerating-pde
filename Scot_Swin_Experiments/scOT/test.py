"""
This script trains a scOT or pretrains Poseidon on a PDE dataset.
Can be also used for finetuning Poseidon.
Can be used in a single config or sweep setup.
"""

import argparse
import torch
import wandb
import numpy as np
import random
import json
import psutil
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
import matplotlib.pyplot as plt
import transformers
from accelerate.utils import broadcast_object_list
from scOT.trainer import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from scOT.model import ScOT, ScOTConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.utils import get_num_parameters, read_cli, get_num_parameters_no_embed
from scOT.metrics import relative_lp_error

from matplotlib.ticker import FormatStrFormatter, FixedLocator  # For colorbar formatting



SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}


def create_predictions_plot(predictions, labels, wandb_prefix):
    assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 4)

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
    )

    vmax, vmin = max(predictions.max(), labels.max()), min(
        predictions.min(), labels.min()
    )

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i % 2 == 0:
            ax.imshow(
                predictions[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    fig.savefig(f"{wandb_prefix}_predictions.png")  # Save locally
    plt.close()


def setup(params, model_map=True):
    config = None
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    CPU_CORES = len(psutil.Process().cpu_affinity())
    # CPU_CORES = min(CPU_CORES, 16)
    CPU_CORES = 1
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=params.wandb_run_name, config=config
        )
        config = wandb.config
    else:

        def clean_yaml(config):
            d = {}
            for key, inner_dict in config.items():
                d[key] = inner_dict["value"]
            return d

        if not params.json_config:
            with open(params.config, "r") as s:
                config = yaml.safe_load(s)
            config = clean_yaml(config)
        run = None

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        if run.sweep_id is not None:
            ckpt_dir = (
                params.checkpoint_path
                + "/"
                + run.project
                + "/"
                + run.sweep_id
                + "/"
                + run.name
            )
        else:
            ckpt_dir = params.checkpoint_path + "/" + run.project + "/" + run.name
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    if model_map and (
        type(config["model_name"]) == str and config["model_name"] in MODEL_MAP.keys()
    ):
        config = {**config, **MODEL_MAP[config["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[config["model_name"]], allow_val_change=True)

    return run, config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scOT or pretrain Poseidon.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Set this to a str pointing to a HF Hub model checkpoint or a directory with a scOT checkpoint if you want to finetune.",
    )
    parser.add_argument(
        "--replace_embedding_recovery",
        action="store_true",
        help="Set this if you have to replace the embeddings and recovery layers because you are not just using the density, velocity and pressure channels. Only relevant for finetuning.",
    )
    parser.add_argument(
        "--plot_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to load the model and plot predictions.",
    )
    params = read_cli(parser).parse_args()
    run, config, ckpt_dir, RANK, CPU_CORES = setup(params)

    train_eval_set_kwargs = (
        {"just_velocities": True}
        if ("incompressible" in config["dataset"]) and params.just_velocities
        else {}
    )
    if params.move_data is not None:
        train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
    if params.max_num_train_time_steps is not None:
        train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
    if params.train_time_step_size is not None:
        train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
    if params.train_small_time_transition:
        train_eval_set_kwargs["allowed_time_transitions"] = [1]
    print("Loading datasets......")
    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    print(f"Train dataset loaded: {len(train_dataset)} samples")
    eval_dataset = get_dataset(
        dataset=config["dataset"],
        which="val",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    print(f"Validation dataset loaded: {len(eval_dataset)} samples")

    #####################################PLot
    # If plot_from_checkpoint is specified, load the checkpoint and skip training
    if params.plot_from_checkpoint:
        print(f"Loading model from checkpoint: {params.plot_from_checkpoint}")
        model = ScOT.from_pretrained(params.plot_from_checkpoint)

        # Create Trainer instance for prediction
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=ckpt_dir,
                per_device_eval_batch_size=config["batch_size"],
                dataloader_num_workers=CPU_CORES,
            ),
            eval_dataset=eval_dataset,
        )

        # Run prediction
        print("Generating predictions...")
        predictions = trainer.predict(eval_dataset)
        
        # Create plots
        print("Creating plots...")
        def create_predictions_plot_with_mask(predictions, labels, dataset, wandb_prefix):
            """
            Create plots masking out hole centers (hole info = 0).

            Args:
                predictions: Predicted output tensor.
                labels: Ground truth labels.
                dataset: Dataset object to extract hole info (third channel).
                wandb_prefix: Prefix for logging plots in WandB.
            """
            # Extract the hole info (third channel) for the first trajectory and timestep
            hole_info = dataset.data[0, 0, ..., 2]  # Shape: (128, 128)
            hole_mask = hole_info == 1  # Mask where hole info is 1

            # Apply the mask to predictions and labels
            masked_predictions = predictions[..., hole_mask].detach().cpu().numpy()
            masked_labels = labels[..., hole_mask].detach().cpu().numpy()

            # Reshape predictions and labels to the masked area for plotting
            reshaped_predictions = masked_predictions.reshape(predictions.shape[0], -1)
            reshaped_labels = masked_labels.reshape(labels.shape[0], -1)

            # Use the original plotting logic
            fig = plt.figure()
            grid = ImageGrid(
                fig, 111, nrows_ncols=(2, reshaped_predictions.shape[0]), axes_pad=0.1
            )

            for idx, ax in enumerate(grid):
                if idx % 2 == 0:  # Even indices are ground truth
                    ax.imshow(
                        reshaped_labels[idx // 2].reshape(hole_mask.shape),
                        cmap="gist_ncar",
                        origin="lower",
                    )
                    ax.set_title("Ground Truth")
                else:  # Odd indices are predictions
                    ax.imshow(
                        reshaped_predictions[idx // 2].reshape(hole_mask.shape),
                        cmap="gist_ncar",
                        origin="lower",
                    )
                    ax.set_title("Prediction")

                ax.set_xticks([])
                ax.set_yticks([])

            # Log the plot to WandB
            save_path = os.path.join(ckpt_dir, "predictions_plot2.png")
            wandb.log({wandb_prefix + "/masked_predictions": wandb.Image(fig)})
            plt.savefig(save_path)
            plt.close(fig)

        def plot_predictions_with_mask(predictions, labels, dataset, mask_channel=2, wandb_prefix="masked_plot", epoch=None):
            """
            Plot predictions and ground truth excluding hole centers using a mask.

            Args:
                predictions: Predicted values (batchsize, channels, height, width).
                labels: Ground truth values (batchsize, channels, height, width).
                dataset: Dataset object to extract hole info (third channel).
                mask_channel: The index of the mask channel in the dataset (default is 2).
                wandb_prefix: Prefix for logging plots in WandB.
                epoch: Current epoch number for naming plots (optional).
            """
            # Extract hole info from the dataset
            hole_info = dataset.data[0, 0, ..., mask_channel]  # Shape: (128, 128)
            hole_mask = hole_info == 1  # Mask where hole info is 1

            # Flatten mask and compute valid coordinates
            coords = np.array(
                [(i, j) for i in range(hole_info.shape[0]) for j in range(hole_info.shape[1])]
            )  # (16384, 2)
            valid_coords = coords[hole_mask.flatten()]  # Only valid points
            valid_indices = hole_mask.flatten()
            example_indices = [100, 400, 800]

            

            # Plotting
            fig, ax = plt.subplots(6, 3, figsize=(18, 30))  # 2x2 grid for both channels
            channels = ["Horizontal Velocity (u)", "Vertical Velocity (v)"]
            cmap = "gist_ncar"
            for i, idx in enumerate(example_indices):
                # Reshape and mask predictions and labels
                predictions_flat = predictions[idx].reshape(predictions.shape[1], -1)
                labels_flat = labels[idx].reshape(labels.shape[1], -1)
                masked_predictions = predictions_flat[:, valid_indices]
                masked_labels = labels_flat[:, valid_indices]
                masked_error = masked_predictions - masked_labels

                for ch in range(len(channels)):
                    row_idx = i*2 + ch
                    truth = masked_labels[ch]
                    pred = masked_predictions[ch]
                    error = masked_error[ch]

                    vmin = min(truth.min(), pred.min())
                    vmax = max(truth.max(), pred.max())
                    error_min, error_max = error.min(), error.max()
                    num_ticks = 5  # Define the number of ticks

                    # Create symmetric ticks
                    max_abs = max(abs(vmin), abs(vmax))
                    symmetric_ticks = np.linspace(-max_abs, max_abs, num_ticks)

                    # Plot ground truth
                    sc1 = ax[row_idx, 0].scatter(
                        valid_coords[:, 0], valid_coords[:, 1], c=truth, cmap=cmap, vmin=vmin, vmax=vmax, s=10
                    )
                    ax[row_idx, 0].set_title(f"Ground Truth - {channels[ch]}")
                    ax[row_idx, 0].axis("off")
                    cb1 = fig.colorbar(sc1, ax=ax[row_idx,0], orientation="vertical", fraction=0.046, pad=0.01, shrink=0.95)
                    cb1.formatter = FormatStrFormatter('%.1f')
                    cb1.locator = FixedLocator(symmetric_ticks)
                    cb1.update_ticks()

                    # Plot predictions
                    sc2 = ax[row_idx, 1].scatter(
                        valid_coords[:, 0], valid_coords[:, 1], c=pred, cmap=cmap, vmin=vmin, vmax=vmax, s=10
                    )
                    ax[row_idx, 1].set_title(f"Prediction - {channels[ch]}")
                    ax[row_idx, 1].axis("off")
                    cb2 = fig.colorbar(sc2, ax=ax[row_idx, 1], orientation="vertical", fraction=0.046, pad=0.01, shrink=0.95)
                    cb2.formatter = FormatStrFormatter('%.1f')
                    cb2.locator = FixedLocator(symmetric_ticks)
                    cb2.update_ticks()

                    # Plot error (prediction - ground truth)
                    sc3 = ax[row_idx, 2].scatter(
                        valid_coords[:, 0], valid_coords[:, 1], c=error, cmap="coolwarm", vmin=error_min, vmax=error_max, s=10
                    )
                    ax[row_idx, 2].set_title(f"Error (Pred - GT) - {channels[ch]}")
                    ax[row_idx, 2].axis("off")
                    cb3 = fig.colorbar(sc3, ax=ax[row_idx, 2], orientation="vertical", fraction=0.046, pad=0.01, shrink=0.95)
                    cb3.formatter = FormatStrFormatter('%.1f')
                    cb3.update_ticks()

            plt.tight_layout()
            plot_title = f"Masked_Plot_Epoch_{epoch}_L_Finetuned_10.png" if epoch else "Masked_Plot.png"
            plt.savefig(plot_title)
            # wandb.log({wandb_prefix: wandb.Image(fig)})
            plt.close(fig)

        def create_predictions_plot_local(predictions, labels, save_path):
            assert predictions.shape[0] >= 4

            indices = random.sample(range(predictions.shape[0]), 4)
            predictions = predictions[indices]
            labels = labels[indices]

            fig = plt.figure()
            grid = ImageGrid(
                fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
            )

            vmax, vmin = max(predictions.max(), labels.max()), min(
                predictions.min(), labels.min()
            )

            for _i, ax in enumerate(grid):
                i = _i // 4
                j = _i % 4

                if i % 2 == 0:
                    ax.imshow(
                        predictions[j, i // 2, :, :],
                        cmap="gist_ncar",
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                    )
                else:
                    ax.imshow(
                        labels[j, i // 2, :, :],
                        cmap="gist_ncar",
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig(save_path)
            plt.close()

        # Save plot locally
        plot_path = os.path.join(ckpt_dir, "predictions_plot.png")
        create_predictions_plot_local(predictions.predictions, predictions.label_ids, plot_path)
        # Assuming predictions and labels are available from validation/testing
        # create_predictions_plot_with_mask(
        #     predictions=predictions.predictions,
        #     labels=predictions.label_ids,
        #     dataset=train_dataset,  # Use the dataset to extract hole info
        #     wandb_prefix="test",   # Adjust prefix as needed
        # )
        plot_predictions_with_mask(
            predictions=predictions.predictions,
            labels=predictions.label_ids,
            dataset=train_dataset,  # Pass your dataset here
            mask_channel=2,  # Assuming 3rd channel holds the hole info
            wandb_prefix="test",
            epoch=100,  # Pass current epoch if available
        )


        print(f"Plots saved to {plot_path}")
        exit()
    print("####################COMPLETE PLOT##########################")