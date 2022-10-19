import cv2
import numpy as np
import serial
#from time import sleep
import time
import sys
#
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

FRAME = None
#

#COM_PORT =   # 請自行修改序列埠名稱
#BAUD_RATES = 9600
ser = serial.Serial('/dev/ttyACM1', 9600)

walk_1 = True
walk_2 = False
catch = False
U_walk = False
water = False
arduino = "aaa"
found_x = False
SEEN = False
turn = False
SEEN_catch= False
LT_num=0
RT_num=0
Rec_num=0
count=10


cap=cv2.VideoCapture(-1)
#cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # 曝光 -4
#cap_side=cv2.VideoCapture(1)
ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
WIDTH = 700
HEIGHT = int(WIDTH/ratio)

lower_red = np.array([0,0,0])  
upper_red = np.array([10,255,255])  
lower_blue = np.array([26,110,51])  
upper_blue = np.array([150,255,182]) 
lower_yellow = np.array([24,81,0])  
upper_yellow = np.array([51,225,255]) 
# lower_black = np.array([0,0,0])  
# upper_black = np.array([179,65,49]) 

lower_red_water = np.array([0,0,0])  
upper_red_water = np.array([10,255,255])  
lower_blue_water = np.array([26,110,51])  
upper_blue_water = np.array([150,255,182]) 
lower_yellow_water = np.array([24,81,0])  
upper_yellow_water = np.array([51,225,255]) 
# lower_black = np.array([0,0,0])  
# upper_black = np.array([179,65,49]) 

#辨認形狀(左 右 方)(無圓)
#顏色(紅黃藍)(無黑)
#水盒對正的顏色
#深度水果
#深度U路
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SHAPE~~~~~~~~~~~~~~~
def shape(img):
    global LT_num
    global RT_num
    global Rec_num
    global count
    # pts = np.array([[100, 100],[500, 100],[500, 500],[100,500],])
    # pts = np.array([pts])
    # mask_ = np.zeros(img.shape[:2], np.uint8)
    # # 在mask上将多边形区域填充为白色
    # cv2.polylines(mask_, pts, 1, 255)    # 描绘边缘
    # cv2.fillPoly(mask_, pts, 255)    # 填充
    # cut = cv2.bitwise_and(img, img, mask=mask_)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^CUT^~~~~~~~~~~~~~~~

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgContour = img.copy()  
    lower_board = np.array([0,0,0])  
    upper_board = np.array([10,255,255]) 
    mask = cv2.inRange(hsv, lower_board, upper_board)
    #result=cv2.bitwise_and(img,img,mask=mask)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations=2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^MASK^~~~~~~~~~~~~~~~
    #print("shaping")
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.circle(imgContour, (300, 212), 5, (2, 255, 75), -1)
    cv2.circle(imgContour, (300, 312), 5, (2, 255, 75), -1)
    cv2.circle(imgContour, (400, 212), 5, (2, 255, 75), -1)
    cv2.circle(imgContour, (400, 312), 5, (2, 255, 75), -1)
    
    try:
        if len(contours)>0 :
            for cnt in contours:
                cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
                M = cv2.moments(opening)  # 尋找質心
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(imgContour, (cX, cY), 5, (213, 255, 75), -1)
                if cX>(WIDTH/2)-50 and cX<(WIDTH/2)+50 and cY>(HEIGHT/2)-50 and cY<(HEIGHT/2)+50: 
                    print("in~!")       
                    # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
                    # cv2.circle(imgContour, (cX, cY), 5, (213, 255, 75), -1)
                    area = cv2.contourArea(cnt)
                    if area > 4000:
                        print(area)
                        peri = cv2.arcLength(cnt, True)
                        vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)
                        corners = len(vertices)  # 算有幾個頂點
                        x, y, w, h = cv2.boundingRect(vertices)
                        cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 4)
                        if RT_num >count:
                            LT_num=0
                            RT_num=0
                            Rec_num=0
                            return("rt",imgContour)
                        if LT_num >count: 
                            LT_num=0
                            RT_num=0
                            Rec_num=0
                            return("left_triangle",imgContour)
                        if Rec_num>count: 
                            LT_num=0
                            RT_num=0
                            Rec_num=0
                            return("rectangle",imgContour)
                        if corners == 3:
                            direction = [cX-vertices[0][0][0], cX -vertices[1][0][0], cX-vertices[2][0][0]]
                            if sum(i > 0 for i in direction) == 1:
                                # if area > 9000:
                                cv2.putText(imgContour, 'left triangle', (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                LT_num = LT_num+1
                                return("no_shape",imgContour)
                                
                            elif sum(i > 0 for i in direction) == 2:
                                # if area > 9000:
                                cv2.putText(imgContour, 'right triangle', (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                RT_num = RT_num+1
                                return("no_shape",imgContour)
                               
                                
                        elif corners == 4:
                            # if area > 9000:
                            cv2.putText(imgContour, 'rectangle', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            Rec_num = Rec_num+1
                            return("no_shape",imgContour)
                        # elif corners >= 5:
                        #     # if area > 9000:
                        #     cv2.putText(imgContour, 'circle', (x, y-5),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            return("no_shape",imgContour)
                    else:
                        return("too_small",imgContour)
                else:
                    return("not_in",imgContour)
        else:
            return("no_contours",img)    
    except: 
        return("nothing",imgContour)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^SHAPE^~~~~~~~~~~~~~~~
def detect(fruit_class, save_img=False, source='', weights='', view_img=True, save_txt=False, imgsz=416, trace=True):
# def detect(save_img=False):
    global FRAME
    source = '-1'
    weights = '/home/toolmen5/yolov7/model_weight/best.pt'
    imgsz = 416
    cuda_device = '0'
    augment_inf = False
    conf_thres = 0.85
    iou_thres = 0.45
    classes = None
    save_conf = False
    agnostic_nms = False
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
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


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
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
                p, s, im0, FRAME = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, FRAME = path, '', im0s, getattr(dataset, 'FRAME', 0)

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
                for *xyxy, conf, cls in reversed(det):
                    if int(cls)==fruit_class:
                        print('class', int(cls), 'position', int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        #claire~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        #fruit_X = int(xyxy[0] + xyxy[2])/2
                        fruit_X = int(xyxy[0] + xyxy[2])/2
                        print("def: ",fruit_X)
                        return(fruit_X)
                        #claire~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # if view_img:  # Add bbox to image
                        #     label = f'{names[int(cls)]} {conf:.2f}'
                        #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            FRAME = im0
            # Stream results
            #if view_img:
                # cv2.imshow(str(p), im0)
                #cv2.waitKey(1)  # 1 millisecond
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|fruit|~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^fruit^~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~WATER~~~~~~~~~~~~~~~
def WaterColor(img,upper,lower):
    imgContour = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    MaxArea = 7000 #也可設成最小值
    CX = 0
    CY = 0
    CNT = 0
    for cnt in contours:
        #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        if area>MaxArea:
            CNT = cnt
            MaxArea = area
            CX = cX
            CY = cY
    try:
        cv2.putText(imgContour, str(MaxArea), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 200), 2)
        cv2.circle(imgContour, (CX, CY), 5, (213, 255, 75), -1)
        cv2.drawContours(imgContour, CNT, -1, (255, 0, 0), 4)
        #print(CX,CY)
        # cv2.imshow('mask', mask)
        # cv2.imshow('imgContour', imgContour)
        #print("Here!")
        return(CX,CY,imgContour)
    except:
        # CX = WIDTH/2
        # CY = HEIGHT/2
        # cv2.putText(imgContour, "nothing", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 200), 2)
        # # cv2.imshow('imgContour', imgContour)
        print("Exception!")
        # return(CX,CY,imgContour)
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~COLOR~~~~~~~~~~~~~~~
def FindColor(img):
    def FindArea(opening):
        img = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img, 150, 200)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if(len(contours)!=0):
            area=0
            for cnt in contours:
                cv2.drawContours(opening, cnt, -1, (0, 255, 255), 4)
                area = cv2.contourArea(cnt) +area
            return (area)
        else:
            return(0)

    pts = np.array([[100, 100],[500, 100],[500, 250],[100,250],])
    pts = np.array([pts])
    mask_ = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(mask_, pts, 1, 255)    # 描绘边缘
    cv2.fillPoly(mask_, pts, 255)    # 填充
    cut = cv2.bitwise_and(img, img, mask=mask_)
    # 裁剪后图像

    hsv = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)
    mask_red=cv2.inRange(hsv,lower_red,upper_red)#過濾顏色紅色
    mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)#過濾顏色藍色
    mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)#過濾顏色黃色
    #mask_black=cv2.inRange(hsv,lower_black,upper_black)#過濾顏色黑色

    result_red=cv2.bitwise_and(img,img,mask=mask_red)
    result_blue=cv2.bitwise_and(img,img,mask=mask_blue)
    result_yellow=cv2.bitwise_and(img,img,mask=mask_yellow)
    #result_black=cv2.bitwise_and(img,img,mask=mask_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening_red = cv2.morphologyEx(result_red, cv2.MORPH_OPEN, kernel, iterations=2)
    opening_blue = cv2.morphologyEx(result_blue, cv2.MORPH_OPEN, kernel, iterations=2)
    opening_yellow = cv2.morphologyEx(result_yellow, cv2.MORPH_OPEN, kernel, iterations=2)
    #opening_black = cv2.morphologyEx(result_black, cv2.MORPH_OPEN, kernel, iterations=2)

    red_area = FindArea(opening_red)
    blue_area = FindArea(opening_blue)
    yellow_area = FindArea(opening_yellow)
    #black_area = FindArea(opening_black)
    
    if (red_area > 10000):
        red_area = FindArea(opening_red)
        cv2.putText(opening_red, str(red_area), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 20), 2)
        cv2.putText(opening_red, "RED!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 20), 2)
        #print("RED!")
        #減速!亮led!等等夾紅番茄~!
        return(opening_red,"red")


    elif (blue_area > 10000):
        blue_area = FindArea(opening_blue)
        cv2.putText(opening_blue, str(blue_area), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 200), 2)
        cv2.putText(opening_blue, "BLUE!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 200), 2)
        return(opening_blue,"blue")
        #print("BLUE!")
        #減速!亮led!等等夾紅番茄~!
        #cv2.imshow("r",opening_blue)

    elif (yellow_area > 10000):
        yellow_area = FindArea(opening_yellow)
        cv2.putText(opening_yellow, str(yellow_area), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 200), 2)
        cv2.putText(opening_yellow, "YELLOW!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 200), 2)
        #print("YELLOW!")
        #減速!亮led!等等夾黃檸檬~!
        return(opening_yellow,"y")
        

    # elif (black_area > 10000):
    #     black_area = FindArea(opening_black)
    #     cv2.putText(opening_black, str(black_area), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 150, 200), 2)
    #     cv2.putText(opening_black, "BLACK!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 200), 2)
    #     return(opening_black,"black")
    
    else :
        return(img,"NOTHING")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^def^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while True:#顯示影片
        #arduino = ser.readline().decode()  # 接收回應訊息並解碼
        ret, frame = cap.read()  # 讀取影片每一幀，cap.read()回傳兩個值，第一個為布林值，取得道的下一幀 
        if ret:                     

                #print("here") 
                img = cv2.resize(frame, (WIDTH, HEIGHT), fx=0.7, fy=0.7)
                img = cv2.flip(img,1)
                if(arduino == "WALK_1"): 
                    walk_1 = True
                    walk_2 = False
                    catch = False
                    U_walk = False
                    water = False
                if(arduino == "WALK_2"): 
                    walk_1 = False
                    walk_2 = True
                    catch = False
                    U_walk = False
                    water = False
                if(arduino == "CATCH"): 
                    walk_1 = False
                    walk_2 = False
                    catch = True
                    U_walk = False
                    water = False
                if(arduino == "U_WALK"): 
                    walk_1 = False
                    walk_2 = False
                    catch = False
                    U_walk = True
                    water = False
                if(arduino == "WATER"): 
                    walk_1 = False
                    walk_2 = False
                    catch = False
                    U_walk = False
                    water = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^buttom^~~~~~~~~                
                while(walk_1 == True) :
                    #arduino = ser.readline().decode()  # 接收回應訊息並解碼
                    result_img_shape = shape(img)[1]
                    Shape = shape(img)[0]
                    #print("walk1")
                    print(Shape)
                    if (Shape == "rt"):
                        ser.write(str.encode("rt"+"\n"))
                        catch = True
                        walk_1 = False
                        time.sleep(0.01)
                    if (Shape == "left triangle"):
                        ser.write(str.encode("left_triangle"+"\n"))
                        time.sleep(0.01)
                        #if(arduino == "TURN_LEFT"):
                            #上斜坡
                    if (Shape == "rectangle"):#要break出walk_1
                        ser.write(str.encode("rectangle"+"\n"))
                        turn = True
                        walk_1 = False
                        time.sleep(0.01)
                    cv2.imshow("Result", result_img_shape)
                    #time.sleep(0.1)
                    #cv2.waitKey(1000)
                    cv2.waitKey(1)
                    break
                while(turn == True):
                    arduino_water = ser.readline()
                    print(arduino_water)
                    if(arduino_water == b'\xffTURN_WATER\n'):
                        water = True
                        turn = False
                        img = cv2.imread("white.png")
                        time.sleep(3)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^walk_1^~~~~~~~~  
                while(catch == True) :
                    catch_color_img = FindColor(frame)[0]
                    color = FindColor(frame)[1]
                    print(color)
                    if(color=="red" or color=="y" or color=="black"):
                        ser.write(str.encode(color+"\n"))
                        while(True):
                            print("arduino")
                            time.sleep(0.01)
                            arduino_catch = ser.readline()
                            #arduino = "SEEN"
                            print("arduino: ",arduino_catch)
                            if(arduino_catch == b'SEEN\n'):
                                SEEN_catch= True
                                break
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~得到顏色接__水果辨識
                    if(SEEN_catch == True):
                        while (True):
                            #ret, frame = cap.read()
                            arduino_catch = ser.readline().decode()  # 接收回應訊息並解碼
                            #
                            #水果辨識
                            #
                            #try:
                            if(color == "red"):fruit_x =detect(2)#tomato
                            if(color == "y"): fruit_x =detect(1)#lemon
                            if(color == "black"): fruit_x =detect(0)#advocado
                                
                            ser.write(str.encode(str(fruit_x)+"\n"))#這樣先強制轉X 過去之後還可以強制轉整數嗎
                            print("fruit_X: ",fruit_x)
                            time.sleep(0.01)
                            arduino_catch_x = ser.readline()
                            if (arduino_catch ==b'GOT_IT\n'):
                                img = cv2.read('white.png')
                                walk_1 = True
                                catch = False
                                break
                            # cv2.imshow("Result", result_img)
                            # cv2.waitKey(1)
                            # break
                    cv2.imshow("Result", catch_color_img)
                    cv2.waitKey(1)
                    break
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^catch^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                while(water == True) :
                    #print("watering")
                    color_img = FindColor(img)[0]
                    color = FindColor(img)[1]
                    print(color)
                    if(color=="red" or color=="y" or color=="blue" or color=="black"):
                        ser.write(str.encode(color+"\n"))
                        while (True):
                            print("arduino")
                            time.sleep(0.01)
                            arduino = ser.readline()
                            #arduino = "SEEN"
                            print("arduino: ",arduino)
                            if(arduino == b'\xffSEEN\n'):
                                SEEN = True
                                break
                            #cv2.imshow("Result", result_img)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~得到顏色接__澆水
                    while(SEEN == True):
                        while(True):
                            ret, frame = cap.read()
                            if (color == "red"):
                                result_img = WaterColor(frame,upper_red,lower_red)[2]
                                water_X=WaterColor(frame,upper_red,lower_red)[0]
                            if(color == "y"):
                                result_img = WaterColor(frame,upper_yellow,lower_yellow)[2]
                                water_X=WaterColor(frame,upper_yellow,lower_yellow)[0]
                            if (color == "blue"):
                                result_img = WaterColor(frame,upper_blue,lower_blue)[2]
                                water_X=WaterColor(frame,upper_blue,lower_blue)[0]
                            # if (color == "black"):
                            #     result_img = WaterColor(frame,upper_black,lower_black)[2]
                            #     water_X=WaterColor(frame,upper_black,lower_black)[0]
                            #     water_Y=WaterColor(frame,upper_black,lower_black)[1]
                            ser.write(str.encode(str(water_X)+"\n"))#這樣先強制轉X 過去之後還可以強制轉整數嗎
                            print("water_X: ",water_X)
                            time.sleep(0.01)
                            arduino_found_x = ser.readline()
                            if(arduino_found_x == b'X_FOUND\n'):
                                found_x = True
                            while(found_x == True):
                                while(True):
                                    ret, frame = cap.read()
                                    if (color == "red"):
                                        result_img = WaterColor(frame,upper_red,lower_red)[2]
                                        water_Y=WaterColor(frame,upper_red,lower_red)[1]
                                    if(color == "y"):
                                        result_img = WaterColor(frame,upper_yellow,lower_yellow)[2]
                                        water_Y=WaterColor(frame,upper_yellow,lower_yellow)[1]
                                    if (color == "blue"):
                                        result_img = WaterColor(frame,upper_blue,lower_blue)[2]
                                        water_Y=WaterColor(frame,upper_blue,lower_blue)[1]
                                    ser.write(str.encode(str(water_Y)+"\n"))    #這樣先強制轉X 過去之後還可以強制轉整數嗎
                                    print("water_Y: ",water_Y)
                                    time.sleep(0.01)
                                    arduino_found_y = ser.readline()  
                                    if(arduino_found_y == b'TURNAROUND\n'):                           
                                        walk_2 = True
                                        water = False
                                        found_x = False
                                        SEEN = False 
                                        img = cv2.read('white.png')
                                        break
                                    # cv2.imshow("Result", result_img)
                                    # cv2.waitKey(1)
                                    break
                            cv2.imshow("Result", result_img)
                            cv2.waitKey(1)
                            break
                            # if cv2.waitKey(1)==ord("q"):#鍵盤上按鍵按下q時，影片中止
                            #     break                           
                    cv2.imshow("Result", color_img)
                    cv2.waitKey(1)
                    break
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^water^~~~~~~~~  
                while(walk_2 == True) :
                    result_img = shape(img)[1]
                    Shape_w2 = shape(img)[0]
                    print("walk_2",Shape_w2)
                    if (Shape_w2 == "rectangle"):#要break出walk_1
                        ser.write(str.encode("rectangle"))
                        U_walk = True
                        walk_2 = False
                    cv2.putText(result_img, "walk_2", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 70, 200), 2)
                    cv2.imshow("Result", result_img)
                    cv2.waitKey(1)
                    break
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^walk_2^~~~~~~~~
                while(U_walk == True) :
                    ##深度U路
                    print("U_walk == True") 
                    time.sleep(100) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^U_walk^~~~~~~~~                      
                # cv2.imshow("Result", result_img)
            



        else:
            break
        
        if cv2.waitKey(1)==ord("q"):#鍵盤上按鍵按下q時，影片中止
            break
