cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED="True"
LOG="../logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

python runner/cpm2_train2.py -i  /media/zal/data/project/Dataset/lsp_dataset --accumulate 6 --num-class 14 -lr 0.00005 --epochs 50 --dataset-type lsp 2>&1 | tee $LOG
