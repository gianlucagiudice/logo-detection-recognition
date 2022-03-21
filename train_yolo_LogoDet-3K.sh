python download_LogoDet-3K.py --dataset_type small
python yolov5/train.py --img 224 --batch 16 --epochs 50 --data LogoDet-3K_small.yaml --weights yolov5/yolov5s.pt
