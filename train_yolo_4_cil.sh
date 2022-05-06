# 1
python train_yolo_4_cil.py --num-class 30 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6
# 2
python train_yolo_4_cil.py --num-class 30 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6 --img-size 320


# 3
python train_yolo_4_cil.py --num-class 100 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6
# 4
python train_yolo_4_cil.py --num-class 100 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m6 --img-size 320


# 5
python train_yolo_4_cil.py --num-class 30 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m
# 6
python train_yolo_4_cil.py --num-class 100 --no-custom-hyper --num-epochs 30 --start-training --yolo-size yolov5m


# 7
python train_yolo_4_cil.py --num-class 30 --adam --custom-hyper --lr 0.01 --num-epochs 30 --start-training
# 8
python train_yolo_4_cil.py --num-class 100 --adam --custom-hyper --lr 0.01 --num-epochs 30 --start-training

# 9
python train_yolo_4_cil.py --num-class 30 --adam --custom-hyper --lr 0.001 --num-epochs 30 --start-training
# 10
python train_yolo_4_cil.py --num-class 100 --adam --custom-hyper --lr 0.001 --num-epochs 30 --start-training
