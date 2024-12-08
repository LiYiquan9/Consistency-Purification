#!/usr/bin/env bash
cd ..

sigma=$1

python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10\
--domain cifar10 \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile results/cifar10-cm-sample_num_10000-noise_$sigma \
--sigma $sigma \
--N 10000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 20 9980) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step \
--checkpoint_dir path_to_checkpoints \
--checkpoint_steps 80 \
--device 0