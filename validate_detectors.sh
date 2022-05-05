echo '----------------- Prepare Dataset -----------------'
python train_yolo_4_cil.py --num-class 100 --no-start-training
echo '------------------------ 1 ------------------------'
python validate_detector.py --name yolo100_cil_classification --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolo-CIL-100cls.pt --cil-model-path weights/resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15
echo '------------------------ 2 ------------------------'
python validate_detector.py --name yolo100_agnostic --single-cls --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolo-CIL-100cls.pt --cil-model-path weights/resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15
echo '------------------------ 3 ------------------------'
python validate_detector.py --name yolo30_cil_classification --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolo-CIL-30cls.pt --cil-model-path weights/resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15
echo '------------------------ 4 ------------------------'
python validate_detector.py --name yolo30_yolo100_agnostic --single-cls --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolo-CIL-30cls.pt --cil-model-path weights/resnet34-pretrained-drop0.5-augmented-onlytop-adam.pt --task test --conf-thres 0.15