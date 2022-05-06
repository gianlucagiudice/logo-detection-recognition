python train_yolo_4_cil.py --num-class 30 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6
python train_yolo_4_cil.py --num-class 100 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6

python train_yolo_4_cil.py --num-class 30 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m
python train_yolo_4_cil.py --num-class 100 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m

python train_yolo_4_cil.py --num-class 30 --adam --custom-hyper --lr 0.01 --num-epochs 30 --start-training
python train_yolo_4_cil.py --num-class 100 --adam --custom-hyper --lr 0.01 --num-epochs 30 --start-training

python train_yolo_4_cil.py --num-class 30 --adam --custom-hyper --lr 0.001 --num-epochs 30 --start-training
python train_yolo_4_cil.py --num-class 100 --adam --custom-hyper --lr 0.001 --num-epochs 30 --start-training
