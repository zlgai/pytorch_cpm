cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED="True"
LOG="../logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

python runner/cpm_train.py -i H:/Dataset/keypoint/lsp/lsp_dataset  --accumulate 6 | $LOG
