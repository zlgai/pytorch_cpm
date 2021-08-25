cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED="True"
LOG="../logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

# python runner/cpm2_train.py -i H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w.csv -v H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w_valid.csv --image-root H:/Project/Github/hrnet_facial_landmark --accumulate 6 --num-class 68 -lr 0.00005 --epochs 4 -o output/face68_2/abc.pch 2>&1 | tee $LOG

python runner/cpm2_train.py -i H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w.csv -v H:/Project/Github/hrnet_facial_landmark/data/hrnet_300w_valid.csv --image-root H:/Project/Github/hrnet_facial_landmark --accumulate 6 --num-class 68 -lr 0.005 --epochs 54 --weights -o output/face68_2/best.pth 2>&1 | tee $LOG

#  --weights H:/Project/Github/openpose_misc/Openpose/weights/pytorch/facenet.pth 
