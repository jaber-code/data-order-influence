#!/bin/bash
#SBATCH --job-name=dataorder
#SBATCH --output=logs/%x_%j/log.out        # Job name and ID in log file
#SBATCH --error=logs/%x_%j/log.err         # Job name and ID in error file
#SBATCH --partition=A100-SDS,A100-PCI,RTXA6000-SDS,RTXA6000,A100-40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=130G
#SBATCH --time=05:00:00

export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}

srun \
	--kill-on-bad-exit \
  	--container-image=/enroot/nvcr.io_nvidia_pytorch_24.11-py3.sqsh \
  	--container-workdir="`pwd`" \
        --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/fscratch/sjaber:/fscratch/sjaber \
	bash -c "pip install torch transformers nltk timm ignite sentence_transformers && \
      	python generator.py"