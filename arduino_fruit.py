import cv2
import numpy as np
import serial
import time
import sys

import cv2
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

ser = serial.Serial('/dev/ttyACM0', 9600)

# load_model return device, model
def load_model(weights = '', imgsz = 416, trace = True):
    weights = 'model_weight/best.pt'
    cuda_device = '0'
    # Initialize
    set_logging()
    device = select_device(cuda_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    return device, model, stride, half

device, model, stride, half = load_model()
print("load success")

def detect(fruit_class, save_img=False, source='', weights='', view_img=True, save_txt=False, imgsz=416, trace=True):
    # save as global variable
    global device
    global model
    global stride
    global half
    source = '0'
    weights = 'model_weight/best.pt' #load_model
    imgsz = 416
    cuda_device = '0' #load_model
    augment_inf = False
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    save_conf = False
    agnostic_nms = False
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
     
    # Set Dataloader
    vid_path, vid_writer = None, None

    # 可不用 if/else
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride) # cv2.capture()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # a: avocado b:lemon c: tomato
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment_inf)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment_inf)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                ser.write(str.encode("start"+"\n"))
                print("start arduino counting")
                for *xyxy, conf, cls in reversed(det):
                    print('class', int(cls), 'position', int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    arduino = ser.readline().decode()
                    if (int(cls)==fruit_class):
                        print("tomato!")
                        print("arduino tomato count: ",arduino)

                    # if(arduino == "3000"):
                    #     print("break")
                    if(arduino == "3000"):
                        break
                    # if cv2.waitKey(1)==ord("q"):
                    #     print("break")
                        #return(int(xyxy[0]))
            #if cv2.waitKey(1)==ord("q"):
                #print("break")
                #break
        
                #~~~~~~~~~~~~~~~~~~~    
            #         if view_img:  # Add bbox to image
            #             label = f'{names[int(cls)]} {conf:.2f}'
            #             plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # # Print time (inference + NMS)
            # # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond
                #!~~~~~~~~~~~~~~~~

# if __name__ == '__main__':
#     detect()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#cap=cv2.VideoCapture(0)
while True:
    #ret, frame = cap.read() 
    
    #if ret : 
    #ser.write(str.encode("start"+"\n"))
    print("start detect")
    detect(2)#tomato       
    print("---------------------------------------------------*")   
        #cv2.imshow("video", frame)
        #detect(2)#tomato
    if cv2.waitKey(1)==ord("q"):
        print("break")
        break