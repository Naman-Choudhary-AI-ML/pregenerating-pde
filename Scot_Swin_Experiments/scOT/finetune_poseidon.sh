#!/bin/bash

# accelerate launch ./scOT/train.py \
#     --config ./configs/run.yaml \
#     --wandb_run_name "scot_Gauss_400_70_45_615_L_finetuned" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy \
#     --finetune_from camlab-ethz/Poseidon-L \
#     --replace_embedding_recovery

# accelerate launch ./scOT/test.py \
#     --config ./configs/run.yaml \
#     --wandb_run_name "scot_Gauss_400_70_45_615_L_finetuned" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy \
#     --plot_from_checkpoint /home/namancho/poseidon/checkpoints/GeoFNO1/scot_Gauss_400_70_45_615_L_finetuned/checkpoint-64971


accelerate launch ./scOT/train.py \
    --config ./configs/run.yaml \
    --wandb_run_name "scot_Gauss_400_100_80_L_pretrained_15_ep" \
    --wandb_project_name "GeoFNO1" \
    --checkpoint_path ./checkpoints/ \
    --data_path /home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy