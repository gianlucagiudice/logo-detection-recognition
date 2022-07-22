echo '------------------------ 1 ------------------------'
echo "Class agnostic"
python validate_detector.py --name time_agnostic_detector --single-cls --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-2993cls_using1000.pt --cil-model-path weights/CIL_1000_250_2993-WA-mem50-resnet34-pretrained-drop0.5-augmented-adam.pt --task test --conf-thres 0.15 --eval-time 1000

echo '------------------------ 2 ------------------------'
echo "Classification mem50 no pruning"
python validate_detector.py --name time_classification_unpruned --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-2993cls_using1000.pt --cil-model-path weights/CIL_1000_250_2993-WA-mem50-resnet34-pretrained-drop0.5-augmented-adam.pt --task test --conf-thres 0.15 --eval-time 1000

echo '------------------------ 3 ------------------------'
echo "Classificationmem50 pruned"
python validate_detector.py --name time_classification_pruning --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-2993cls_using1000.pt --cil-model-path weights/PRUNED_spars1_CIL_1000_250_2993-noWA-mem50-resnet34-pretrained-drop0.3-augmented-adam --task test --conf-thres 0.15 --eval-time 1000

echo '------------------------ 4 ------------------------'
echo "Classification mem50 KD"
python validate_detector.py --name time_classification_kd --data LogoDet-3K_CIL.yaml --yolo-model-path weights/yolov5m6-CIL-512px-2993cls_using1000.pt --cil-model-path weights/CIL_1000_250_2993-WA-mem50-resnet34-pretrained-drop0.5-augmented-adam.pt --student-model-path weights/kd_resnet50-drop0.3-mem50.ckpt --task test --conf-thres 0.001  --eval-time 1000
