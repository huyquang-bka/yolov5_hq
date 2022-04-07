import os
import sys
from pathlib import Path
import opt
import time

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

from utils.general import xyxy2xywh


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


source = str(opt.source)
# Load model
device = select_device(opt.device)
model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

# Half
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.model.half() if opt.half else model.model.float()

dt, seen = [0.0, 0.0, 0.0], 0
check = -1


def detect(img0):
    im0 = img0.copy()
    img = letterbox(img0, opt.imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=opt.augment, visualize=opt.visualize)

    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            id_dict = {}
            spot_dict = {}
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            xyxys, confs, clss = [], [], []
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # cv2.imwrite(f"LP_image/{x1}_{y1}_{x2}_{y2}.jpg", im0[y1:y2, x1:x2])
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im0


if __name__ == '__main__':
    path = r"C:\Users\Admin\Downloads\test_LP.mp4"
    cap = cv2.VideoCapture(path)
    frame = 0
    with torch.no_grad():
        while True:
            frame += 1
            t = time.time()
            ret, img0 = cap.read()
            img0 = detect(img0)
            cv2.imshow("Image", img0)
            # vid_writer.write(img0)
            key = cv2.waitKey(1)
            print("FPS: ", 1 // (time.time() - t))
            if key == ord("q"):
                break
