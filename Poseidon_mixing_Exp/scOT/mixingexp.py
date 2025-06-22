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
    CPU_CORES = min(CPU_CORES, 16)
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=params.wandb_run_name, mode="online", config=config
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
    if RANK in (0, -1):
        base = params.checkpoint_path.rstrip("/")
        proj = run.project if run is not None else params.wandb_project_name
        name = None
        if run is not None and run.sweep_id:
            # inside a sweep
            name = f"{run.sweep_id}/{run.name}"
        else:
            name = params.wandb_run_name
        ckpt_dir = params.checkpoint_path
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
    parser.add_argument("--alpha", type=float, default=None,
                        help="(array) override the alpha in the YAML")
    parser.add_argument("--finetune_from", type=str, default=None,
                        help="Checkpoint directory or HF Hub model for finetuning")
    parser.add_argument("--total_trajectories", type=int, default=None,
                        help="Override total number of trajectories")
    parser.add_argument("--replace_embedding_recovery", action="store_true",
                        help="Replace embedding & recovery layers when finetuning")
    parser.add_argument("--plot_from_checkpoint", type=str, default=None,
                        help="Load a trained model and just plot predictions")

    # ⇓ *** add your new options here, BEFORE read_cli() touches the parser ***
    parser.add_argument("--num_easy", type=int, default=None,
                        help="Number of easy (no‑hole) trajectories in TRAIN set")
    parser.add_argument("--num_hard", type=int, default=None,
                        help="Number of hard (hole) trajectories in TRAIN set")

    # let the helper extend the *same* parser or return a modified one
    parser = read_cli(parser)          # many helpers return a new parser
    params = parser.parse_args()       # now all flags are recognised

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

    # alpha = config.get("alpha")
    if params.num_easy is not None and params.num_hard is not None:
        num_no_hole_train = params.num_easy
        num_hole_train    = params.num_hard
        num_finetune_samples = num_no_hole_train + num_hole_train
        config["num_trajectories"] = num_finetune_samples
        # make it visible in WandB even offline
        if run is not None:
            wandb.config.update(
                {"num_easy": num_no_hole_train,
                 "num_hard": num_hole_train},
                allow_val_change=True)
    if run is not None:
        # tell wandb about it, even in offline mode
        wandb.config.update({"alpha": params.alpha}, allow_val_change=True)
    # alpha = config.get("alpha")
    #Define the number of samples
    # num_finetune_samples = config["num_trajectories"]
    # alpha_to_total = {
    #     0.50:   2,
    #     0.02:   50,
    #     0.01:  100,
    #     0.005:  200,
    #     0.0025:  400,
    #     0.00125:  800,
    #     0.00063:   1600,  # if alpha=1 still means only complex, so total=1
    #     0.00031:  3200,
    #     0.00016:  6400
    # }

    # pick the total based on alpha (fall back to YAML default if missing)
    if params.total_trajectories is not None:
        total_trajs = params.total_trajectories
    else:
        # fallback to config default
        total_trajs = config["num_trajectories"]
    config["num_trajectories"] = total_trajs
    num_finetune_samples = total_trajs
    num_val_samples = 100
    num_test_samples = 80

    #Compute dataset splits
    # num_hole_train = int(alpha * num_finetune_samples)
    # num_no_hole_train = num_finetune_samples - num_hole_train
    #── NEW SPLIT ── always fix hole to 1, let no-hole fill out the rest
    # num_hole_train    = 1 if total_trajs >= 1 else 0
    # num_no_hole_train = num_finetune_samples - num_hole_train
    num_hole_val = num_val_samples // 2
    num_no_hole_val = num_val_samples - num_hole_val

    print(f"Finetuning dataset: {num_hole_train} hole + {num_no_hole_train} no-hole")
    print(f"Validation dataset: {num_hole_val} hole + {num_no_hole_val} no-hole")

    def streaming_stats(path, block=100):
        # mmap the file so we never load it all at once
        arr = np.load(path, mmap_mode="r")[..., :3]            # (Ntraj, T, H, W, 3)
        s, ss, cnt = np.zeros(3), np.zeros(3), 0
        for i in range(0, arr.shape[0], block):
            chunk = arr[i : i+block]                           # (b, T, H, W, 3)
            s  += chunk.sum(axis=(0,1,2,3))
            ss += (chunk**2).sum(axis=(0,1,2,3))
            cnt += np.prod(chunk.shape[:4])                    # ← include width too
        return s, ss, cnt

    # hole
    s1, ss1, c1 = streaming_stats(config["hole_data_path"])
    # no-hole
    s2, ss2, c2 = streaming_stats(config["no_hole_data_path"])

    total_sum   = s1  + s2
    total_sqsum = ss1 + ss2
    total_count = c1  + c2

    global_mean = total_sum / total_count
    global_std  = np.sqrt(total_sqsum/total_count - global_mean**2)

    # Load datasets with updated N_val and N_test
    if num_hole_train != 0 :
        train_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="train",
            num_trajectories=num_hole_train,
            N_val=num_hole_val,
            N_test=num_test_samples,
            data_path=config["hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    if num_no_hole_train != 0 :
        train_no_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="train",
            num_trajectories=num_no_hole_train,
            N_val=num_no_hole_val,
            N_test=num_test_samples,
            data_path=config["no_hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
        
    # Load validation datasets
    if num_hole_train != 0 :
        val_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="val",
            num_trajectories=num_hole_train,  #to keep the validation dataset same across experiments
            N_val=num_hole_val,
            N_test=num_test_samples,
            data_path=config["hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    else:
        val_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="val",
            num_trajectories=1,  #to keep the validation dataset same across experiments
            N_val=num_hole_val,
            N_test=num_test_samples,
            data_path=config["hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    if num_no_hole_train != 0 :
        val_no_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="val",
            num_trajectories=num_no_hole_train,  #to keep the validation dataset same across experiments
            N_val=num_no_hole_val,
            N_test=num_test_samples,
            data_path=config["no_hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    else:
        val_no_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="val",
            num_trajectories=1,  #to keep the validation dataset same across experiments
            N_val=num_no_hole_val,
            N_test=num_test_samples,
            data_path=config["no_hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    if num_hole_train == 0:
        train_dataset = train_no_hole_dataset
    elif num_no_hole_train == 0:
        train_dataset = train_hole_dataset
    else:
        # Combine hole and no-hole datasets for training
        train_dataset = train_hole_dataset + train_no_hole_dataset
        
    eval_dataset = val_hole_dataset + val_no_hole_dataset

    print(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(eval_dataset)}")
    print(f"Val data split is for hole: {len(val_hole_dataset)}, and no hole: {len(val_no_hole_dataset)}")

    config["effective_train_set_size"] = len(train_dataset)
    time_involved = isinstance(train_dataset, BaseTimeDataset) or (
        isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and isinstance(train_dataset.datasets[0], BaseTimeDataset)
    )

    if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
        resolution = train_dataset.resolution
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        channel_slice_list = train_dataset.channel_slice_list
        printable_channel_description = train_dataset.printable_channel_description
    else:
        resolution = train_dataset.datasets[0].resolution
        input_dim = train_dataset.datasets[0].input_dim
        output_dim = train_dataset.datasets[0].output_dim
        channel_slice_list = train_dataset.datasets[0].channel_slice_list
        printable_channel_description = train_dataset.datasets[
            0
        ].printable_channel_description
    # output_dim = 2
    model_config = (
        ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels= output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=channel_slice_list,
            residual_model="convnext",
            use_conditioning=time_involved,
            learn_residual=False,
        )
        if params.finetune_from is None or params.replace_embedding_recovery
        else None
    )
    print(f"Expected number of channels in input: {input_dim}")
    print(f"Expected number of channels in output: {output_dim}")


    train_config = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        evaluation_strategy="epoch",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_accumulation_steps=16,
        max_grad_norm=config["max_grad_norm"],
        num_train_epochs=config["num_epochs"],
        optim="adamw_torch",
        learning_rate=config["lr"],
        learning_rate_embedding_recovery=(
            None
            if (params.finetune_from is None or "lr_embedding_recovery" not in config)
            else config["lr_embedding_recovery"]
        ),
        learning_rate_time_embedding=(
            None
            if (params.finetune_from is None or "lr_time_embedding" not in config)
            else config["lr_time_embedding"]
        ),
        weight_decay=config["weight_decay"],
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        log_level="passive",
        logging_strategy="steps",
        logging_steps=5,
        logging_nan_inf_filter=False,
        save_strategy= "epoch",
        save_total_limit=1,
        seed=SEED,
        fp16=False,
        dataloader_num_workers=CPU_CORES,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        auto_find_batch_size=False,
        full_determinism=False,
        torch_compile=False,
        report_to=[],
        run_name=params.wandb_run_name,
        disable_tqdm = False
    )
    print("TrainingArguments setup:")
    print(f"Batch size: {train_config.per_device_train_batch_size}")
    print(f"Epochs: {train_config.num_train_epochs}")
    print(f"Learning rate: {train_config.learning_rate}")


    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=0.0,  # set no threshold for now
    )

    if params.finetune_from is not None:
        model = ScOT.from_pretrained(
            params.finetune_from, config=model_config, ignore_mismatched_sizes=True
        )
        print("FINETUNED MODEL LOADED SUCCESSFULLY")
    else:
        model = ScOT(model_config)
    print("Model initialized successfully.")
    print(f"Model size: {get_num_parameters(model)} parameters")
    print(f"Model size without embeddings: {get_num_parameters_no_embed(model)}")
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(f"Model size: {num_params}")
        print(f"Model size without embeddings: {num_params_no_embed}")

    def compute_metrics(eval_preds):
        channel_list = channel_slice_list

        def get_statistics(errors, metric_type):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                f"median_relative_{metric_type}_error": median_error,
                f"mean_relative_{metric_type}_error": mean_error,
                f"std_relative_{metric_type}_error": std_error,
                f"min_relative_{metric_type}_error": min_error,
                f"max_relative_{metric_type}_error": max_error,
            }
        #Compute L1 errors (existing functionality)
        error_statistics_l1 = [
            get_statistics(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                ),
                metric_type="l1"
            )
            for i in range(len(channel_list) - 1)
        ]

        #Compute L2 errors
        error_statistics_l2 = [
            get_statistics(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i]:channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i]:channel_list[i + 1]],
                    p=2,
                    return_percent=True,
                ),
                metric_type="l2"
            )
            for i in range(len(channel_list) - 1)
        ]
        #Compute Linfi errors
        error_statistics_linf = [
            get_statistics(
                np.max(
                    np.abs(
                        eval_preds.predictions[:, channel_list[i]:channel_list[i + 1]] -
                        eval_preds.label_ids[:, channel_list[i]:channel_list[i + 1]]
                    ).reshape(eval_preds.predictions.shape[0], eval_preds.predictions.shape[1], -1),
                    axis=-1 #takeing max over spacial
                ).max(axis = -1),
                metric_type="linf"
            )
            for i in range(len(channel_list) - 1)
        ]


        if output_dim == 1:
            # error_statistics = error_statistics[0]
            return {**error_statistics_l1[0], **error_statistics_l2[0], **error_statistics_linf[0]}
            # return error_statistics
        
        else:
            mean_over_means_l1 = np.mean(
                np.array(
                    [stats["mean_relative_l1_error"] for stats in error_statistics_l1]
                ),
                axis = 0,
            )
            mean_over_means_l2 = np.mean(
                np.array(
                    [stats["mean_relative_l2_error"] for stats in error_statistics_l2]
                ),
                axis = 0,
            )
            mean_over_means_linf = np.mean(
                np.array(
                    [stats["mean_relative_linf_error"] for stats in error_statistics_linf]
                ),
                axis = 0,
            )
            mean_over_medians_l1 = np.mean(
                np.array(
                    [stats["median_relative_l1_error"] for stats in error_statistics_l1]
                ),
                axis = 0,
            )
            mean_over_medians_l2 = np.mean(
                np.array(
                    [stats["median_relative_l2_error"] for stats in error_statistics_l2]
                ),
                axis = 0,
            )
            mean_over_medians_linf = np.mean(
                np.array(
                    [stats["median_relative_linf_error"] for stats in error_statistics_linf]
                ),
                axis = 0,
            )
            error_statistics_ = {
                "mean_relative_l1_error": mean_over_means_l1,
                "mean_over_median_relative_l1_error": mean_over_medians_l1,
                "mean_relative_l2_error": mean_over_means_l2,
                "mean_over_median_relative_l2_error": mean_over_medians_l2,
                "mean_relative_linf_error": mean_over_means_linf,
                "mean_over_median_relative_linf_error": mean_over_medians_linf,
            }
            for i, (stats_l1, stats_l2, stats_linf) in enumerate(zip(error_statistics_l1, error_statistics_l2, error_statistics_linf)):
                for key, value in stats_l1.items():
                    error_statistics_[f"{printable_channel_description[i]}/{key}"] = value
                for key, value in stats_l2.items():
                    error_statistics_[f"{printable_channel_description[i]}/{key}"] = value
                for key, value in stats_linf.items():
                    error_statistics_[f"{printable_channel_description[i]}/{key}"] = value
            return error_statistics_

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    print("Starting training loop...")
    trainer.train(resume_from_checkpoint=params.resume_training)
    print("Training loop completed.")
    trainer.save_model(train_config.output_dir)

    torch.cuda.empty_cache()

    print("Testing on HOLE dataset...")
    if num_hole_train != 0 :
        test_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=num_hole_train,
            N_val=num_hole_val,
            N_test=num_test_samples,
            data_path=config["hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs

        )
    else:
        test_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=1,
            N_val=num_hole_val,
            N_test=num_test_samples,
            data_path=config["hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs

        )
    predictions_hole = trainer.predict(test_hole_dataset, metric_key_prefix="test_hole")
    print(f"[test_hole] metrics: {predictions_hole.metrics}", flush=True)
    wandb.log(predictions_hole.metrics)
    torch.cuda.empty_cache()

    print("Testing on NO HOLE dataset...")
    # Load test datasets separately
    if num_no_hole_train != 0 :
        test_no_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=num_no_hole_train,
            N_val=num_no_hole_val,
            N_test=num_test_samples,
            data_path=config["no_hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )
    else:
        test_no_hole_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=1,
            N_val=num_no_hole_val,
            N_test=num_test_samples,
            data_path=config["no_hole_data_path"],
            mean       = global_mean,
            std        = global_std,
            **train_eval_set_kwargs
        )

    predictions_no_hole = trainer.predict(test_no_hole_dataset, metric_key_prefix="test_no_hole")
    wandb.log(predictions_no_hole.metrics)
    print(f"[test_no_hole] metrics: {predictions_no_hole.metrics}", flush=True)
    
    # wandb.log({f"test_no_hole/mean_relative_l1_error": predictions_no_hole.metrics["mean_relative_l1_error"],
    #         f"test_no_hole/mean_relative_l2_error": predictions_no_hole.metrics["mean_relative_l2_error"],
    #         f"test_no_hole/mean_relative_linf_error": predictions_no_hole.metrics["mean_relative_linf_error"]})

    
    # wandb.log({f"test_hole/mean_relative_l1_error": predictions_hole.metrics["mean_relative_l1_error"],
    #         f"test_hole/mean_relative_l2_error": predictions_hole.metrics["mean_relative_l2_error"],
    #         f"test_hole/mean_relative_linf_error": predictions_hole.metrics["mean_relative_linf_error"]})

    print("Testing completed. All results logged to WandB under separate test categories.")
