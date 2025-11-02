#!/bin/bash
#SBATCH --job-name=UniInstruct
#SBATCH --account=cw220
#SBATCH --partition=commons
#SBATCH --time=1-00:00:00    
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=48         
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=200G                  
#SBATCH --mail-user=yw251@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out 

source /scratch/cw220/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/yw251/conda/envs/uni_instruct

#Reproduce Uni_Instruct distillation of pretrained EDM models

# Retrieve the dataset name from the first argument
dataset=$1

# Example usage:
# To set specific GPUs and run the script for 'cifar10-uncond':
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# sh run_uni_instruct.sh 'cifar10-uncond'

# Tip: Decrease --batch-gpu to reduce memory consumption on limited GPU resources

if [ "$dataset" = 'cifar10-uncond' ]; then
    # Command to execute the Uni-Instruct training script with specified parameters
    # Optional: Use the --resume option to load a specific checkpoint, e.g.:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    # If --resume points to a folder, the script will automatically load the latest checkpoint from that folder. 
    # This is particularly useful for seamless resumption when running the code in a cluster environment.
    # Note: Optional parameters, such as --data_stat, will be computed automatically within the code if not explicitly provided.
    torchrun --standalone --nproc_per_node=2 fsim_train.py \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 128 \
    --data './checkpoints/cifar10-32x32.zip'  \
    --outdir './image_experiment/fsim-train-runs/cifar10-uncond-forward-clean' \
    --divergence 'Forward-Clean' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model cifar10-uncond \
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
    --data_stat '/gpfs/share/home/2206192113/cvpr_code/Uni-Instruct/checkpoints/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full,is50k \
    --save_best_and_last 1 

    #--sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl'

elif [ "$dataset" = 'cifar10-cond' ]; then
    torchrun --standalone --nproc_per_node=4 fsim_train.py \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 32 \
    --data './datasets/cifar10-32x32.zip'  \
    --outdir './image_experiment/fsim-train-runs/cifar10-cond-f-distill-jensen-shannon' \
    --divergence 'f-distill-Jensen-Shannon' \
    --variance_reduction 1 \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model '/ailab/user/wangyifei/SiD-main/checkpoints/edm-cifar10-32x32-cond-vp.pkl' \
    --detector_url '/ailab/user/wangyifei/SiD-main/checkpoints/inception-2015-12-05.pt' \
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
    --data_stat '/ailab/user/wangyifei/SiD-main/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    # --resume './image_experiment/fsim-train-runs/cifar10-cond' 
    #--sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/cifar10-cond/alpha1.2/network-snapshot-1.200000-713312.pkl'

    
elif [ "$dataset" = 'imagenet64-cond' ]; then
    torchrun --standalone --nproc_per_node=4 fsim_train.py \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 8 \
    --data '/ailab/user/wangyifei/SiD-main/datasets/imagenet-64x64.zip' \
    --outdir './image_experiment/fsim-train-runs/imagenet64-cond-chi-square' \
    --divergence 'Chi-Square' \
    --nosubdir 0 \
    --arch adm \
    --edm_model "/ailab/user/wangyifei/SiD-main/checkpoints/edm-imagenet-64x64-cond-adm.pkl" \
    --detector_url '/ailab/user/wangyifei/SiD-main/checkpoints/inception-2015-12-05.pt' \
    --tick 20 \
    --snap 50 \
    --dump 200 \
    --lr 4e-6 \
    --glr 4e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 300 \
    --data_stat '/ailab/user/wangyifei/SiD-main/imagenet-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 300 
    #--sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl'

elif [ "$dataset" = 'cifar10-uncond-sid' ]; then
    # Command to execute the SiDA training script with specified parameters
    # Optional: Use the --resume option to load a specific checkpoint, e.g.:
    # --resume 'image_experiment/sid-train-runs/cifar10-uncond/training-state-????.pt'
    # If --resume points to a folder, the script will automatically load the latest checkpoint from that folder. 
    # This is particularly useful for seamless resumption when running the code in a cluster environment.
    # Note: Optional parameters, such as --data_stat, will be computed automatically within the code if not explicitly provided.
    torchrun --standalone --nproc_per_node=2 fsim_train.py \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 128 \
    --data './checkpoints/cifar10-32x32.zip'  \
    --outdir './image_experiment/fsim-train-runs/cifar10-uncond-forward-kl' \
    --resume "/gpfs/share/home/2301111469/yifei/Uni-Instruct/image_experiment/fsim-train-runs/cifar10-uncond-forward-kl/00001-cifar10-32x32-SiD-SiDA-uncond-ddpmpp-edm-glr1e-05-lr1e-05-ls1.0_lsg100.0_lsd1.0_lsg_gan0.01-initsigma2.5-gpus2-batch256-tmax800-fp32_403968batchgpu32/training-state-022528.pt" \
    --divergence 'Forward-KL' \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model './checkpoints/edm-cifar10-32x32-uncond-vp.pkl' \
    --detector_url './checkpoints/inception-2015-12-05.pt' \
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
    --duration 100 \
    --data_stat './checkpoints/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full,is50k \
    --save_best_and_last 1 \
    --sid_model '/gpfs/share/home/2301111469/yifei/Uni-Instruct/checkpoints/cifar-uncond-network-snapshot-1.200000-403968.pkl'

elif [ "$dataset" = 'cifar10-cond-sid' ]; then
    torchrun --standalone --nproc_per_node=2 fsim_train.py \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 64 \
    --data './checkpoints/cifar10-32x32.zip'  \
    --outdir './image_experiment/fsim-train-runs/cifar10-cond-forward-kl' \
    --divergence 'Forward-KL' \
    --resume "/gpfs/share/home/2301111469/yifei/Uni-Instruct/image_experiment/fsim-train-runs/cifar10-cond-forward-kl/00005-cifar10-32x32-SiD-SiDA-cond-ddpmpp-edm-glr1e-05-lr1e-05-ls1.0_lsg100.0_lsd1.0_lsg_gan0.01-initsigma2.5-gpus2-batch256-tmax800-fp32_713312batchgpu64/training-state-045056.pt" \
    --variance_reduction 0 \
    --nosubdir 0 \
    --arch ddpmpp \
    --edm_model './checkpoints/edm-cifar10-32x32-cond-vp.pkl' \
    --detector_url './checkpoints/inception-2015-12-05.pt' \
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
    --data_stat './checkpoints/cifar10-32x32.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --sid_model /gpfs/share/home/2301111469/yifei/Uni-Instruct/checkpoints/network-snapshot-1.200000-713312.pkl
    # --resume './image_experiment/fsim-train-runs/cifar10-cond' 

    
elif [ "$dataset" = 'imagenet64-cond-sid' ]; then
    torchrun --standalone --nproc_per_node=4 fsim_train.py \
    --cond 1 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 8192 \
    --batch-gpu 8 \
    --data '/ailab/user/wangyifei/SiD-main/datasets/imagenet-64x64.zip' \
    --outdir './image_experiment/fsim-train-runs/imagenet64-cond-forward-kl' \
    --divergence 'Forward-KL' \
    --nosubdir 0 \
    --arch adm \
    --edm_model "/ailab/user/wangyifei/SiD-main/checkpoints/edm-imagenet-64x64-cond-adm.pkl" \
    --detector_url '/ailab/user/wangyifei/SiD-main/checkpoints/inception-2015-12-05.pt' \
    --tick 20 \
    --snap 50 \
    --dump 200 \
    --lr 4e-6 \
    --glr 4e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --lsd 1 \
    --lsg_gan 0.01 \
    --duration 300 \
    --data_stat '/ailab/user/wangyifei/SiD-main/imagenet-64x64.npz' \
    --use_gan 1 \
    --metrics fid50k_full \
    --save_best_and_last 1 \
    --dropout 0.1 \
    --augment 0 \
    --ema 2 \
    --duration 300 \
    --sid_model 'https://huggingface.co/UT-Austin-PML/SiD/resolve/main/imagenet64/alpha1.2/network-snapshot-1.200000-939176.pkl'
    
else
    echo "Invalid dataset specified"
    exit 1
fi
