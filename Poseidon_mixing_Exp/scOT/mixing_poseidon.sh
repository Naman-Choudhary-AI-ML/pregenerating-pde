#!/bin/bash

# accelerate launch ./scOT/train.py \
#     --config ./configs/run.yaml \
#     --wandb_run_name "Irr_FT_PL_scot_Gauss_400_100_80_finetuned" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy \
#     --finetune_from camlab-ethz/Poseidon-L \
#     --replace_embedding_recovery

#Finetuning script
#data path here is not used anywhere in the script, but the argument is still needed so put something
accelerate launch ./scOT/mixingexp.py \
    --config ./configs/mixing.yaml \
    --wandb_run_name "FT_1.0_alpha_LDCmixingexp_nohole_vs_centre_hole_air_PosT_400_100_80" \
    --wandb_project_name "GeoFNO1" \
    --checkpoint_path /data/user_data/namancho/checkpoints_poseidon \
    --data_path /home/namancho/datasets/FPO_ext_NS_irreg/final_dataset.npy \
    --finetune_from camlab-ethz/Poseidon-T \
    --replace_embedding_recovery

# accelerate launch ./scOT/train.py \
#     --config ./configs/mixing.yaml \
#     --wandb_run_name "FT_LDCmixingexp_air_PosT_400_100_80" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path /data/user_data/namancho/checkpoints_poseidon \
#     --data_path /home/namancho/datasets/FPO_ext_NS_irreg/final_dataset.npy \
#     --finetune_from camlab-ethz/Poseidon-T \
#     --replace_embedding_recovery
# accelerate launch ./scOT/test.py \
#     --config ./configs/run.yaml \
#     --wandb_run_name "scot_Gauss_400_70_45_615_L_finetuned" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/NS-Gauss-Irr-Openfoam/sliced_data1200.npy \
#     --plot_from_checkpoint /home/namancho/poseidon/checkpoints/GeoFNO1/scot_Gauss_400_70_45_615_L_finetuned/checkpoint-64971

#Pretraining Script

# accelerate launch ./scOT/train.py \
#     --config ./configs/run.yaml \
#     --wandb_run_name "Reg_LDC_air_PosB_400_100_80" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/LDC-Openfoam/OpenFOAM_final.npy

# accelerate launch ./scOT/train.py \
#     --config ./configs/run_reg.yaml \
#     --wandb_run_name "Reg_scot_Gauss_400_100_80_L_pretrained_15_ep" \
#     --wandb_project_name "GeoFNO1" \
#     --checkpoint_path ./checkpoints/ \
#     --data_path /home/namancho/datasets/NS-Gauss-Openfoam/openfoam1200.npy