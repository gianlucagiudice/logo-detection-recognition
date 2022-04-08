python download_LogoDet-3K.py --dataset-type normal --sampling-fraction 1 --only-top --train-split 0.70 --validation-split 0.10 --test-split 0.20
python yolov5/train.py --img 512 --batch 16 --epochs 50 --data LogoDet-3K.yaml --weights yolov5/yolov5m6.pt
