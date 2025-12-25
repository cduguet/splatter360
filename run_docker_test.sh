#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate splat360
output_dir='./outputs/real_data_experiment_docker/'
checkpoint_path='./checkpoints/hm3d.ckpt'

python -m src.main \
    +experiment=real_data \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type='log_depth' \
    output_dir=$output_dir \
    dataset.near=0.1 \
    dataset.overfit_to_scene='real_scene' \
    checkpointing.load=$checkpoint_path \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path='assets/evaluation_index_real.json' \
    test.eval_depth=true \
    mode='test' \
    test.compute_scores=true