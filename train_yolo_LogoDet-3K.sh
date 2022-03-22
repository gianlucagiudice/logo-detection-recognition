python download_LogoDet-3K.py --dataset_type normal --sampling_fraction 0.25
python yolov5/train.py --img 224 --batch 16 --epochs 50 --data LogoDet-3K.yaml --weights yolov5/yolov5s.pt
