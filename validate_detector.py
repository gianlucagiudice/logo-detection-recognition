import logging
from threading import Thread

import pandas as pd
import torch.utils.cpp_extension
from PIL import Image
from torchvision import transforms
import pickle

from config import DATASET_PATH, LOGODET_3K_NORMAL_PATH, METADATA_CROPPED_IMAGE_PATH

from yolov5.val import *

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # pycil

sys.path.append(str(ROOT / 'pycil'))
from pycil.utils.inc_net import DERNet
from pycil.utils.transformations import iLogoDet3K_trsf
from pycil.teacher_student import TeacherStudent


def init_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def load_cil_model(cil_model_path, student_model_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if student_model_path:
        teacher_model, idx2class, class2idx, pred_folder = load_cil_model(cil_model_path, False)
        lightning_dict = torch.load(student_model_path, map_location=device)
        teacher_student_model = TeacherStudent(teacher_model, lightning_dict['arch'],  lightning_dict['args'])
        teacher_student_model.load_state_dict(lightning_dict['state_dict'])

        final_model = teacher_student_model

    else:
        model_dict = torch.load(cil_model_path, map_location=device)

        cil_model = DERNet(model_dict['convnet_type'], model_dict['pretrained'], model_dict['dropout_rate'])

        for n_classes in np.cumsum(model_dict['task_sizes']):
            cil_model.update_fc(n_classes)

        try:
            cil_model.load_state_dict(model_dict['state_dict'])
        except RuntimeError:
            from torch.nn import Conv2d, BatchNorm2d
            original_layers = dict([(key, value) for key, value in cil_model.named_modules() if
                                    isinstance(value, Conv2d) or isinstance(value, BatchNorm2d)])
            for key, value in original_layers.items():
                if isinstance(value, Conv2d):
                    value.weight.data = model_dict['state_dict'][f'{key}.weight']
                elif isinstance(value, BatchNorm2d):
                    value.weight.data = model_dict['state_dict'][f'{key}.weight']
                    value.bias.data = model_dict['state_dict'][f'{key}.bias']
                    value.running_mean.data = model_dict['state_dict'][f'{key}.running_mean']
                    value.running_var.data = model_dict['state_dict'][f'{key}.running_var']
            cil_model.load_state_dict(model_dict['state_dict'])

        n_classes = len(model_dict['class_remap'])
        assert len(model_dict['cil_class2idx']) == n_classes
        assert len(model_dict['cil_idx2class']) == n_classes
        final_model = cil_model
        idx2class, class2idx, pred_folder = model_dict['cil_idx2class'], model_dict['cil_class2idx'],  model_dict['cil_prediction2folder']

    final_model.eval()
    return final_model, idx2class, class2idx, pred_folder


def load_cil_image(path, xyxy):
    # Unzip prediction
    xyxy = np.array(xyxy.round().cpu(), dtype=int)
    # Read the image
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    # Crop image
    img = img.crop(xyxy)
    # Image transformation
    common_trsf = iLogoDet3K_trsf['common']
    test_trsf = iLogoDet3K_trsf['test']
    all_trsf = transforms.Compose([*test_trsf, *common_trsf])
    img = all_trsf(img)
    return img


def resolve_labels(labelsn, cropped2metadata, full2cropped_list, names2id, paths, si):
    nl = labelsn.shape[0]

    descriptions_gt = [cropped2metadata[x] for x in full2cropped_list[Path(paths[si]).name]]
    labels_gt = list(torch.tensor([l['native_label'] for l in descriptions_gt]).float())
    brands_gt = [x['brand'] for x in descriptions_gt]
    labels_yolo = labelsn[:, 1:].round().cpu()
    resolved_labels = torch.zeros(nl, 1).cpu()
    # Resolve labels
    for i, yolo in enumerate(labels_yolo):
        # Get the nearest label
        d = torch.tensor([((yolo.cpu() - gt) ** 2).sum() for gt in labels_gt]).cpu()
        resolved = d.argmin().item()
        # Resolve labels
        resolved_labels[i] = names2id[brands_gt[resolved]]
        # Remove the resolved label
        labels_gt.pop(resolved)
        brands_gt.pop(resolved)

    return resolved_labels


def get_test_instances(metadata, test_instances):
    metadata_test = metadata.loc[
        metadata['new_path'].map(lambda x: x.name).isin(test_instances)
    ]

    print('Building index . . .')
    metadata_test_dict = {}
    cropped2metadata = {}
    for i, row in tqdm(metadata_test.iterrows(), total=len(metadata_test)):
        cropped_img, _, full_img, brand, _, yolo_label, native_label = row
        cropped_img, full_img = cropped_img.name, full_img.name
        metadata_test_dict[full_img] = metadata_test_dict.get(full_img, []) + [cropped_img]
        cropped2metadata[cropped_img] = {
            'full_img': full_img,
            'brand': brand,
            'yolo_label': yolo_label,
            'native_label': native_label
        }

    print('Creating dict . . .')
    full2cropped_list = {}
    n_cropped_test_instances = 0
    all_cropped_test = set()
    for full_img in tqdm(metadata_test['new_path'].unique(), total=metadata_test['new_path'].unique().size):
        with open(Path(DATASET_PATH) / 'LogoDet-3K_det4cil' / 'labels' / full_img.with_suffix('.txt')) as f:
            labels = [x.strip() for x in f.readlines()]
            n_cropped_test_instances += len(labels)

        filtered_instances_cropped = set()
        for cropped_img in metadata_test_dict[full_img.name]:
            if cropped2metadata[cropped_img]['yolo_label'] in labels:
                filtered_instances_cropped.add(cropped_img)
                all_cropped_test.add(cropped_img)
        full2cropped_list[full_img.name] = filtered_instances_cropped

    assert len(cropped2metadata) == len(metadata_test)
    assert len(all_cropped_test) == n_cropped_test_instances
    assert len(full2cropped_list) == len(test_instances)

    return metadata_test_dict, cropped2metadata, full2cropped_list, all_cropped_test


@torch.no_grad()
def run(
        metadata,
        test_instances,
        data,
        cil_model_path=None,
        student_model_path=None,
        yolo_model_path=None,  # model.pt path(s)
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.15,  # confidence threshold
        iou_thres=0.5,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=12,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=True,  # augmented inference
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        project=ROOT / 'validation_res',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        # Eval time
        eval_time=0
):

    # Initialize/load model and set device
    training = model is not None

    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Init logger
    LOGGER = init_logger(save_dir / 'detection_results.log')

    # Load model
    model = DetectMultiBackend(ROOT / yolo_model_path, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    # Data
    data = check_dataset(data)  # check

    # Configure
    cil_model, cil_idx2class, cil_class2idx, cil_class_remap = load_cil_model(
        ROOT / cil_model_path, student_model_path)
    model.eval()
    cuda = device.type != 'cpu'

    if single_cls:
        nc = 1
    else:
        (metadata_test_dict, cropped2metadata,
         full2cropped_list, all_cropped_test) = get_test_instances(metadata, test_instances)

        # Unique classes
        unique_classes = {cropped2metadata[cropped]['brand'] for cropped in all_cropped_test}
        assert len(unique_classes) == len(cil_class_remap)
        nc = len(unique_classes)

    print(f'Number of unique classes: {nc}')

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if eval_time:
        batch_size = 1

    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres)
    s = ('%30s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')

    print(f'Batch_size: {dataloader.batch_size}')

    if eval_time:
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=eval_time)  # progress bar
    else:
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    eval_time_i = eval_time
    dt_list = []
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        dt_dict = {}

        if eval_time and eval_time_i <= 0:

            with open(save_dir / 'times.pickle', 'wb') as f:
                pickle.dump(dt_list, f)

            print('End')
            exit(0)

        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        dt_dict['convert_img'] = t2 - t1

        # Inference
        t_temp = time_sync()
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        t_temp_final = time_sync() - t_temp
        dt[1] += time_sync() - t2

        dt_dict['yolo_inference'] = t_temp_final

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        t_temp = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=True)
        t_temp_final = time_sync() - t_temp
        dt[2] += time_sync() - t3

        dt_dict['non_max_supp'] = t_temp_final

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]

            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                if not single_cls:
                    # Resolve labels
                    resolved_labels = resolve_labels(
                        labelsn, cropped2metadata, full2cropped_list, cil_class2idx, paths, si)
                    resolved_labels = resolved_labels.to(device)
                    # Assign resolved labels
                    labelsn[:, 0:1] = resolved_labels
                    # Replace labels in targets (for the plots)
                    targets[(targets[:, 0] == si).nonzero(as_tuple=False), 1] = resolved_labels

                    # ------- Classification of ROIs -------
                    t4 = time_sync()

                    # Make predictions
                    predictions = torch.zeros(predn.shape[0], 1)
                    for i, prediction in enumerate(predn):
                        xyxy = prediction[:4]
                        img = load_cil_image(paths[si], xyxy)
                        cil_prediction = cil_model(img.expand(1, *img.shape))
                        if not student_model_path:
                            cil_prediction = cil_prediction['logits']
                        logit_argmax = cil_prediction.argmax().int().item()
                        predictions[i] = cil_class_remap[logit_argmax]

                    predn[:, 5:6] = predictions

                    dt_dict['cil_inference'] = time_sync() - t4

                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)

            # Copy labels
            pred[:, 5] = predn[:, 5]
            labels[:, 0] = labelsn[:, 0]
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            callbacks.run('on_val_image_end', pred, predn, path, cil_idx2class, im[si])

        # Plot images
        if plots and batch_i < 10:
            if single_cls:
                names = {k: 'logo' for k, _ in cil_idx2class.items()}
            else:
                names = cil_idx2class
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

        callbacks.run('on_val_batch_end')

        dt_list.append(dt_dict)
        if eval_time:
            eval_time_i = eval_time_i - 1

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=cil_idx2class)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    LOGGER.info(s)
    pf = '%30s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if len(stats) and nc > 1:
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (cil_idx2class[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(cil_idx2class.values()))
        callbacks.run('on_val_end')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    # Save validation dict
    res_dict = {
        'mean_precision': mp,
        'mean_recall': mr,
        'map50': map50,
        'map': map,
        'ap_classes': maps,
        'times': t,
        'id2name': cil_idx2class
    }

    path = save_dir / 'validation.pickle'
    with open(path, 'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (mp, mr, map50, map), maps, t, cil_idx2class


def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment name
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # Dataset
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # Models
    parser.add_argument('--yolo-model-path', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--cil-model-path', type=str, required=True, help='CIL model path')
    parser.add_argument('--student-model-path', type=str, required=False, help='Student Model of KD')
    # Detector config
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # Detector hyper-parameters
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')

    # Detector hyper-parameters
    parser.add_argument('--eval-time', type=int, default=None, help='evaluation time')

    # Other
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt


def main(opt):
    # Metadata df
    metadata = pd.read_pickle(Path(DATASET_PATH) / LOGODET_3K_NORMAL_PATH / METADATA_CROPPED_IMAGE_PATH)
    # Test images full format
    with open(Path(DATASET_PATH) / 'LogoDet-3K_det4cil' / 'test.txt') as f:
        test_instances = [Path(x.strip()).name for x in f.readlines()]

    # Run validation
    run(metadata, test_instances, **vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
