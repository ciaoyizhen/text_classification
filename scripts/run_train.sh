source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1,2,3
torchrun --nproc-per-node=3 main.py configs/test.yaml
