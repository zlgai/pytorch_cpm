cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED="True"
LOG="../logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

python runner/cpm_train.py -i H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w.csv -v H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w_valid.csv --image-root H:/Project/Github/hrnet_facial_landmark --accumulate 6 --num-class 68 --dataset-type csv   2>&1 | tee $LOG
#  --weights H:/Project/Github/openpose_misc/Openpose/weights/pytorch/facenet.pth 
