# Consistency Purification: Mitigating Diffusion Purification Trade-offs towards Certified Robustness #

## Prepare Environment

- Install Pytorch:
    ```bash
    conda create -n consistency_purification python=3.9
    conda activate consistency_purification
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 
    ```
- Install Dependecies:
    ```bash
    pip install -r requirements.txt
    pip install --upgrade jaxlib==0.4.7+cuda12.cudnn88 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    conda install -c conda-forge mpi4py
    ```

## Datasets and Pre-trained Models
CIFAR-10 dataset and ViT-B/16 model on CIFAR-10 will be downloaded automatically.

For DDPM-onestep experiment, please download model with this link and place it under pretrained directory:  
   [cifar10_uncond_50M_500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)

For EDM-onestep and EDM_multistep experiment, please download model with this link and place it under consistency_models_cifar10/EDM/checkpoints, and name it as checkpoint_40:
    [edm_cifar10_ema](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/edm_cifar10_ema)
  
For Consistency Purification experiment, please download model with this link and place it under consistency_models_cifar10/CD-LPIPS/checkpoints:
    [cd-lpips](https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/cd-lpips/checkpoints/checkpoint_80)

For Consistency Purification with Consistency Fine-tuning experiment, please download model with this link and place it under consistency_models_cifar10/CD-LPIPS/checkpoints:
    [cd-lpips-finetune](https://huggingface.co/datasets/YiquanLi/Consistency_Purification/blob/main/checkpoint_120)


## Consistency Fine-tuning
To run Consistency Fine-tuning on pre-trained Consistency Model:
```
cd consistency_models_cifar10
python -m jcm.main --config configs/cifar10_ve_cd.py --workdir CD-LPIPS --mode train --config.optim.lr=0.0001 --config.training.loss_norm='lpips'
```

## Run Experiments 

By default, sigma will be 0.25/0.5/1.0 for CIFAR-10.

To get certified accuracy of DDPM onestep:
```
cd run_scripts
bash carlini22_cifar10.sh [sigma]
```

To get certified accuracy of EDM onestep:
```
cd run_scripts
bash edm_cifar10_onestep.sh [sigma]
```

To get certified accuracy of EDM multistep:
```
cd run_scripts
bash edm_cifar10_multistep.sh [sigma]
```

To get certified accuracy of Consistency Purification (For certified accuracy of Consistency Purification with Consistency Fine-tuning, replace the checkpoint of finetuned consistency model):
```
cd run_scripts
bash cm_cifar10_multistep.sh [sigma]
```

## Check Results
The results will be stored under results directory. To compute the certified accuracy, run:
```
cd results
bash compute_accuracy.py [path_to_result_file]
```
