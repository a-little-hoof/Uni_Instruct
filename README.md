# Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction

         
> Yifei Wang, Weimin Bai, Colin Zhang, Debing Zhang, Weijian Luo, He Sun       
> *NeurIPS 2025 ([arXiv 2505.20755](https://arxiv.org/abs/2505.20755))*  

## Contact 

Feel free to contact us if you have any questions about the paper!

- Yifei Wang [yw251@rice.edu](mailto:yw251@rice.edu)
- Weijian Luo [pkulwj1994@icloud.com](mailto:pkulwj1994@icloud.com)

## Abstract

In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the *Uni-Instruct*. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and *1.38* for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step teacher diffusion with a significant improvement margin of 1.33 (1.02 vs 2.35). We also apply Uni-Instruct on broader tasks like text-to-3D generation. For text-to-3D generation, Uni-Instruct gives decent results, which slightly outperforms previous methods, such as SDS and VSD, in terms of both generation quality and diversity. Both the solid theoretical and empirical contributions of Uni-Instruct will potentially help future studies on one-step diffusion distillation and knowledge transferring of diffusion models.


## Environment Setup
```
conda env create -f environment.yaml
conda activate uni_instruct
```

## Prepare Dataset
Follow the instructions of [EDM](https://github.com/NVlabs/edm). We also provided a Google Drive version, see below.

*Important!* We split imagenet512-sd.zip(~150GB) into 20 subsets to avoid OOM problem.

First, run shell script:
```
bash split.sh
``` 
After that, before training, you might need to adjust the dataset path at line 997 and line 344.

## Training Script
The training script is [run_fsim.sh](https://github.com/a-little-hoof/Uni-Instruct/blob/main/run_fsim.sh) 
Here is an example command, there are a few lines you should replace with your own path.
```
if [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 fsim_train.py \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data './datasets/cifar10-32x32.zip'  \
    --outdir './image_experiment/fsim-train-runs/cifar10-cond-chi-square' \
    --divergence 'Chi-Square' \
    --resume "/ailab/user/wangyifei/fsim/image_experiment/fsim-train-runs/cifar10-cond-chi-square/00000-cifar10-32x32-SiDA-cond-ddpmpp-edm-glr1e-05-lr1e-05-ls1.0_lsg100.0_lsd1.0_lsg_gan0.01-initsigma2.5-gpus4-batch256-tmax800-fp32batchgpu32/training-state-024576.pt" \ ### resume previous experiments, you can simply remove this line if you're training from scratch.
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model '/ailab/user/wangyifei/SiD-main/checkpoints/edm-cifar10-32x32-cond-vp.pkl' \ ### pretrained model, downloaded from EDM
    --detector_url '/ailab/user/wangyifei/SiD-main/checkpoints/inception-2015-12-05.pt' \ ### pretrained model that is used to calculate FID and IS, also downloaded from EDM 
    --tick 10 \
    --snap 50 \
    --dump 200 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 300 \
    --data_stat '/ailab/user/wangyifei/SiD-main/cifar10-32x32.npz' \ ### data statistics, downloaded from EDM
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
```
Update the following 5 parameters: 
```
--resume ... (delete this line)
--outdir PATH/TO/THE/DIRECTORY/OF/THE/OUTPUTS \
--edm_model 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl' \
--detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
--data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
```
Here are several divergences that the code supported:
```
--divergence "Forward-KL" \
--divergence "Reverse-KL" \
--divergence "Jeffrey-KL" \
--divergence "Chi-Square" \
```
We also reimplement the divergence mentioned in f-distill, which is an integral version of our distillation method:
```
--divergence "f-distill-Forward-KL" \
--divergence "f-distill-Reverse-KL" \
--divergence "f-distill-Jensen-Shannon" \
```

## Model Weights and Datasets
- ImageNet64 dataset: [ImageNet 64*64](https://drive.google.com/file/d/1UYnWH40Ed9uSWzl6fdXpim33MO7uzluk/view?usp=sharing).
- CIFAR10 pretrained EDM model: [EDM-cifar10-cond](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl)
- ImageNet64 pretrained EDM model: [EDM-ImageNet64-cond](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl)

## Resume training
To resume training from previous experiments, download the .pt files below and add --resume argument to the training script:
```
--resume /PATH/TO/THE/DOWNLOADED/.PT/FILE \
```
Resume the training process of Forward-KL, Reverse-KL, and jeffrey-KL on ImageNet64: [Model Weights](https://disk.pku.edu.cn/link/AA1C01BF2D551748748927920652F8C5B2).

## Acknowledgements
Uni-Instruct is built upon SiDA and EDM. We extend our gratitude to the authors of SiDA paper and EDM paper for sharing their code. The EDM repository can be found here: [NVlabs/edm](https://github.com/NVlabs/edm/). The SiDA repository can be found here: [mingyuanzhou/SiD](https://github.com/mingyuanzhou/SiD/tree/sida).

