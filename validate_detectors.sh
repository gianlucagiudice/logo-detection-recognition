echo '----------------- Prepare Dataset -----------------'
python train_yolo_4_cil.py --num-class 2993 --no-start-training --init-cls 1000 --increment-cls 250
echo '------------------------ 1 ------------------------'
python validate_detector.py --name CIL_2993 --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-1000cls.pt --cil-model-path weights/CIL_1000_250_2993-mem100-resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15
echo '------------------------ 2 ------------------------'
python validate_detector.py --name CIL_2993_agnostic --single-cls --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-1000cls.pt --cil-model-path weights/CIL_1000_250_2993-mem100-resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15
