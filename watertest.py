import cv2
import numpy as np
import serial
from time import sleep

ser = serial.Serial('COM17', 9600)
motorOutput = "000000\n"
taskOutput = "900000\n"
motorOrTask = True  #decide to send which output to arduino (true motor/ false task)
variables = dict.fromkeys(['signCode', 'signCounter', 'colorCode', 'colorCounter'], 0)
friutName = ['tomato', 'lemon', 'blue', 'avocado']
queryfruit = ''
pixelcount = []
Iterm = 0
for i in range(640):
    pixelcount.append(0)
tubeQue = []
for i in range(70):
    tubeQue.append('128128\n')


#                       sign            pot             tube           red          yellow          blue            black
maskLowwerBound = [[  0,166,158], [  0,117,  0], [   0,125,  0], [  0,188,  0], [ 12,184, 91], [ 86,  0,243], [  0,  0,  0]]
maskUpperBound  = [[  8,255,255], [  9,223,186], [ 179,255,255], [ 22,255,255], [ 31,255,192], [117,192,255], [179, 95,164]]
maskName = dict.fromkeys(['signMask', 'potMask', 'tubeMask', 'redSideMask', 'yellowSideMask', 'blueSideMask', 'blackSideMask', 'redWaterMask', 'yellowWaterMask', 'blueWaterMask', 'blackWaterMask', 'potShow', 'signShow', 'tubeShow']) 
state = 0

def maskAll():  #process masks  input: three caps/ output: eleven masked img (sign, pot, tube, 4 colors)
    # frontHsv=cv2.cvtColor(frontFrame,cv2.COLOR_BGR2HSV)
    # sideHsv=cv2.cvtColor(sideFrame,cv2.COLOR_BGR2HSV)
    waterHsv=cv2.cvtColor(waterFrame,cv2.COLOR_BGR2HSV)
    # for i in range(3):  #mask sign, pot, tube
    #     mask = cv2.inRange(frontHsv,np.array(maskLowwerBound[i]),np.array(maskUpperBound[i]))
    #     maskName[list(maskName)[i]] = cv2.bitwise_and(frontFrame,frontFrame,mask=mask)
    # maskName['signMask'] = cv2.cvtColor(maskName['signMask'],cv2.COLOR_BGR2GRAY)
    # for i in range(3, 7): #mask side four color
    #   mask = cv2.inRange(sideHsv,np.array(maskLowwerBound[i]),np.array(maskUpperBound[i]))
    #   maskName[list(maskName)[i]] = cv2.bitwise_and(sideHsv,sideHsv,mask=mask)
    for i in range(7, 11):    #mask water four color
      mask = cv2.inRange(waterHsv,np.array(maskLowwerBound[i-4]),np.array(maskUpperBound[i-4]))
      maskName[list(maskName)[i]] = cv2.bitwise_and(waterFrame, waterFrame,mask=mask)
      maskName[list(maskName)[i]] = cv2.cvtColor(maskName[list(maskName)[i]],cv2.COLOR_BGR2GRAY)

def signDetect():   #input: mask sign img/ output: (int)variables['signCode'] (0: None, 1: left tri, 2: right tri, 3: square, 4: circle) and return True
    contours, hierarchy = cv2.findContours(maskName['signMask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 回傳兩個值，第一個值是輪廓，第二個值是階層
    # x,y,w,h=-1,-1,-1,-1
    for cnt in contours:
        # print(cnt)#輪廓點的數字,顯示在run的結果
        area = cv2.contourArea(cnt)  # 輪廓面積
        cv2.drawContours(maskName['signShow'], cnt, -1, (0, 255, 0), 3)
        if area > 15000:  # 扣除雜訊
            peri = cv2.arcLength(cnt, True)  # 輪廓邊長(輪廓,輪廓是否閉合)
            vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)  # 用多邊形近似輪廓(輪廓,近似值,輪廓是否閉合)會回傳多邊形頂點
            corners = len(vertices)  # 有幾個頂點
            x, y, w, h = cv2.boundingRect(vertices)  # 回傳四個值:左上角的x、左上角的y值、寬度、高度
            cv2.putText(maskName['signShow'], str(area), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.drawContours(maskName['signShow'], cnt, -1, (0, 255, 0), 3)  # (畫在哪,要畫的輪廓,要畫的輪廓是第幾個-1表每一個都畫,顏色,粗度)
            if corners==3:#是個三角形
                #cv2.putText(maskName['signShow'],"triangle",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                newapp = np.ravel(vertices)
                flag_y, flag_x, sec_y, sec_x = 0, 0, 0, 0
                for i in range(1, len(newapp), 2):
                    if newapp[i] > flag_y:
                        sec_y = flag_y
                        sec_x = flag_x
                        flag_y = newapp[i]
                        flag_x = newapp[i - 1]
                    elif newapp[i] > sec_y:
                        sec_y = newapp[i]
                        sec_x = newapp[i - 1]
                if flag_x >= sec_x:
                    cv2.putText(maskName['signShow'], "left", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    variables['signCode'] = '1'
                    cv2.imshow('sign', maskName['signShow'])
                    return True
                elif flag_x <= sec_x:
                    cv2.putText(maskName['signShow'], "right", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    variables['signCode'] = '2'
                    cv2.imshow('sign', maskName['signShow'])
                    return True
            elif corners == 4: #是個正方形
                cv2.putText(maskName['signShow'], "rectangle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                variables['signCode'] = '3'
                cv2.imshow('sign', maskName['signShow'])
                return True
            elif corners == 8:  # 是個正方形
                cv2.putText(maskName['signShow'], "circle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                variables['signCode'] = '4'
                cv2.imshow('sign', maskName['signShow'])
                return True
    cv2.putText(maskName['signShow'], "None", (10, 480-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('sign', maskName['signShow'])
    return False

def potDetect():    #input: mask pot img/ output: (change global variable) motorOutput
    global motorOutput
    global taskOutput
    turnratio = 0.3
    firstcoor = -1
    upbound = 180
    lowbound = 280
    boundary = [0, 0, 0, 0]
    for i in range(0, 640):
        pixel = 0
        pixelcount[i] = 0
        num = np.uint8(0)
        #chop the image
        for j in range(upbound, lowbound):
            if (maskName['potMask'][j][i][0] != num or maskName['potMask'][j][i][1] != num or maskName['potMask'][j][i][2] != num): 
                pixel += 1
        if pixel > 25:
            pixelcount[i] = pixel
        #clear if continuous density too small
            if firstcoor == -1:
                firstcoor = i
        elif firstcoor != -1:
            if i - firstcoor < 10:
                for j in range(firstcoor, i):
                    pixelcount[j] = 0
            firstcoor = -1
        #find range for left and right
        if i < 320:
            if pixelcount[i] != 0:
                if boundary[0] == 0:
                    boundary[0] = i
                else:
                    boundary[1] = i
        elif i >= 320:
            if pixelcount[i] != 0:
                if boundary[2] == 0:
                    boundary[2] = i
                else:
                    boundary[3] = i
    leftdiff = boundary[1] - boundary[0]
    rightdiff = boundary[3] - boundary[2]
    moveratio = (leftdiff-rightdiff)*turnratio
    #show density
    for i in range(640):
        if pixelcount[i] != 0:
            maskName['potShow'] = cv2.line(maskName['potShow'], (i, 480), (i, 480-pixelcount[i]), (255, 255, 0), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[0], 480), (boundary[0], 0), (0, 0, 255), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[1], 480), (boundary[1], 0), (0, 0, 255), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[2], 480), (boundary[2], 0), (0, 0, 255), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[3], 480), (boundary[3], 0), (0, 0, 255), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (0, upbound), (640, upbound), (0, 0, 255), 1)
            maskName['potShow'] = cv2.line(maskName['potShow'], (0, lowbound), (640, lowbound), (0, 0, 255), 1)
    if abs(leftdiff-rightdiff) < 25:
        cv2.putText(maskName['potShow'], "go straight", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    elif leftdiff > rightdiff:
        cv2.putText(maskName['potShow'], "turn right", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(maskName['potShow'], "turn left", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    left = 128 + moveratio 
    right = 128 - moveratio
    left_str = str(int(left))
    right_str = str(int(right))
    if right < 100:
        right_str = '0' + right_str
    if left < 100:
        left_str = '0' + left_str
    motorOutput = left_str + right_str +'\n'
    print('boundary:', boundary)
    print('leftdiff:', leftdiff, 'rightdiff:', rightdiff)
    print(motorOutput)
    cv2.imshow('pot', maskName['potShow'])
    # cv2.imshow('pot', maskName['potShow'])

def colorDetect():  #input: four color mask img/ output: (change global variable) colorSign and return True
    global queryfruit
    for i in range(4):
        maskName[list(maskName)[i+3]] = cv2.cvtColor(maskName[list(maskName)[i+3]],cv2.COLOR_HSV2BGR)
        maskName[list(maskName)[i+3]] = cv2.cvtColor(maskName[list(maskName)[i+3]],cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(maskName[list(maskName)[i+3]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 7000:  
                variables['colorCode'] = i
                queryfruit = friutName[i]
                return True
    return False
    # maskName[list(maskName)[variables['colorCode']]] = cv2.cvtColor(maskName[list(maskName)[variables['colorCode']]],cv2.COLOR_GRAY2BGR)

def fruitDetect():  #input: four color mask img/ output: loop until get fruit and return True
    fruitFrame = cv2.resize(sideFrame, (0, 0), fx=1.2, fy=1.2)
    cv2.imwrite('fruitFrame.jpg',fruitFrame)
    prediction = model.predict('fruitFrame.jpg', confidence=40, overlap=30)
    # cv2.imshow('video', frame)
    # infer on a local image
    print(prediction.json())
    print(prediction)
    print(prediction.plot())

    for i in range (len(prediction)) :
        if(prediction[i]["class"] == queryfruit):
            x = prediction[i]["x"]
            y = prediction[i]["y"]
            width = prediction[i]["width"]
            height = prediction[i]["height"]
            volume = width*height
            print('x:', x, ' y:',y)
            print('volume:', volume)

def waterDetect():  #input: four color mask img/ output: loop until water
    contours, hierarchy = cv2.findContours(maskName[list(maskName)[variables['colorCode']+7]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    global motorOutput
    global taskOutput
    global motorOrTask
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 7000:
            cv2.drawContours(waterFrame, cnt, -1, (128, 0, 128), 3)
            x, y, w, h = cv2.boundingRect(cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True))
            cv2.circle(waterFrame, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
            if x+w/2 < 300:
                cv2.putText(waterFrame, "motor forward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
                motorOutput = '050050\n'
            elif x+w/2 > 340:
                cv2.putText(waterFrame, "motor backward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
                motorOutput = '850850\n'
            else:
                if y+h/2 < 220:
                    cv2.putText(waterFrame, "stepper forward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
                    taskOutput = '902000\n'
                    motorOrTask = False
                elif y+h/2 > 260:
                    cv2.putText(waterFrame, "stepper backward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
                    taskOutput = '901000\n'
                    motorOrTask = False
                else:
                    cv2.putText(waterFrame, "water!!", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
                    taskOutput = '900100\n'
                    motorOrTask = False
                    return True
        cv2.rectangle(waterFrame, (300,220), (340, 260), (255, 255, 255), 3)
    # maskName[list(maskName)[variables['colorCode']]] = cv2.cvtColor(maskName[list(maskName)[variables['colorCode']]],cv2.COLOR_GRAY2BGR)
    cv2.imshow('result', waterFrame)
    return False

def tubeDetect():   #input: mask tube img/ output: (change global variable) motorOutput
    def tubeDetect(): #input: mask tube img/ output: (change global variable) motorOutput
 global motorOutput
 global tubeQue
 global Iterm
 Iratio, turnratio, luCounter, llCounter ,ruCounter, rlCounter = 0.4, 0.8, 0, 0, 0, 0
 upbound = 360
 lowbound = 380
 boundary = [0, 0, 0, 0]
 num = np.uint8(0)
 for i in range(0, 290):
  #chop the image
  if boundary[0] == 0:
   if (maskName['tubeMask'][upbound][i][0] != num or maskName['tubeMask'][upbound][i][1] != num or maskName['tubeMask'][upbound][i][2] != num):
    if luCounter > 2:
     boundary[0] = i
     cv2.circle(maskName['tubeShow'], (i, upbound), 5, (255, 204, 0), -1)
    else :
     luCounter += 1
   else:
    luCounter = 0
  if boundary[1] == 0:
   if (maskName['tubeMask'][lowbound][i][0] != num or maskName['tubeMask'][lowbound][i][1] != num or maskName['tubeMask'][lowbound][i][2] != num):
    if llCounter > 2:
     boundary[1] = i
     cv2.circle(maskName['tubeShow'], (i, lowbound), 5, (255, 204, 0), -1)
    else :
     llCounter += 1
   else:
    llCounter = 0
 for i in range(350, 640):
  num = np.uint8(0)
  #chop the image
  if boundary[2] == 0:
   if (maskName['tubeMask'][upbound][i][0] != num or maskName['tubeMask'][upbound][i][1] != num or maskName['tubeMask'][upbound][i][2] != num):
    if ruCounter > 2:
     boundary[2] = i
     cv2.circle(maskName['tubeShow'], (i, upbound), 5, (255, 204, 0), -1)
    else :
     ruCounter += 1
   else:
    ruCounter = 0
  if boundary[3] == 0:
   if (maskName['tubeMask'][lowbound][i][0] != num or maskName['tubeMask'][lowbound][i][1] != num or maskName['tubeMask'][lowbound][i][2] != num):
    if rlCounter > 2:
     boundary[3] = i
     cv2.circle(maskName['tubeShow'], (i, lowbound), 5, (255, 204, 0), -1)
    else :
     rlCounter += 1
   else:
    rlCounter = 0
 if (boundary[0] == 0 and boundary[1] != 0):
  if boundary[1] > 160:
   boundary[0] = 320
 if (boundary[1] == 0 and boundary[0] != 0):
  if boundary[0] > 160:
   boundary[1] = 320
 if (boundary[2] == 0 and boundary[3] != 0):
  if boundary[3] > 480:
   boundary[2] = 640
  else:
   boundary[2] = 320
 if (boundary[3] == 0 and boundary[2] != 0):
  if boundary[2] > 480:
   boundary[3] = 640
  else:
   boundary[3] = 320
  
 leftdiff = abs(boundary[1] - boundary[0])
 rightdiff = abs(boundary[3] - boundary[2])
 timediff = 0.1
 Iterm += (leftdiff-rightdiff) * timediff
 moveratio = (leftdiff-rightdiff)*turnratio + Iterm *  Iratio
 cv2.line(maskName['tubeShow'], (0, upbound), (640, upbound), (0, 0, 255), 1)
 cv2.line(maskName['tubeShow'], (0, lowbound), (640, lowbound), (0, 0, 255), 1)
 if abs(leftdiff-rightdiff) < 20:
  cv2.putText(maskName['tubeShow'], "go straight", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
 elif leftdiff > rightdiff:
  cv2.putText(maskName['tubeShow'], "turn right", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
 else:
  cv2.putText(maskName['tubeShow'], "turn left", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
 left = 128 + moveratio 
 right = 128 - moveratio
 left_str = str(int(left))
 right_str = str(int(right))
 if right < 100:
  right_str = '0' + right_str
 if left < 100:
  left_str = '0' + left_str
 now = left_str + right_str +'\n'
 tubeQue.append(now)
 motorOutput = tubeQue.pop(1)
 cv2.imshow('tube', maskName['tubeShow'])
 print('boundary:', boundary)
 print('leftdiff:', leftdiff, 'rightdiff:', rightdiff)
 print('now',now)
 print('motorOutput', motorOutput)

def potAndSign(currentState):
    global state
    global taskOutput
    global motorOrTask
    potDetect()
    if signDetect():
        # cv2.imshow('sign', maskName['signShow'])
        if variables['signCounter'] > 3:        # counter to be tested
            motorOrTask = False
            taskOutput = '9'+variables['signCode']+'0000\n'
            state = currentState + 1
            print('change state to', state)
            variables['signCounter'] = 0
        else:
            variables['signCounter'] += 1
    else:
        variables['signCounter'] = 0
        # cv2.imshow('sign', maskName['signShow'])

def potAndColor(currentState):
    global state
    potDetect()
    if colorDetect():
        if variables['colorCounter'] > 3:
            state = 1 + currentState
            variables['colorCounter'] = 0
            print('find color', queryfruit)
        else:
            variables['colorCounter'] += 1
    else:
        variables['colorCounter'] = 0

def switch():
    global state
    if state == 0:
        potAndSign(state)
    elif state == 1:
        potAndColor(state)
    elif state == 2:
        potDetect()
        if fruitDetect():
            state = 3
    elif state == 3:
        potAndSign(state)
    elif state == 4:
        potAndSign(state)
    elif state == 5:
        potAndColor(state)
    elif state == 6:
        potDetect()
        if waterDetect():
            state = 7
    elif state == 7:
        potAndSign(state)
    elif state == 8:
        potAndSign(state)
    elif state == 9:
        tubeDetect()

# frontCap = cv2.VideoCapture(0)
# sideCap = cv2.VideoCapture(1)
waterCap = cv2.VideoCapture(0)
ser.write(motorOutput.encode('utf-8'))
# serdiff = True
while True:
    while ser.in_waiting:

        serinput = int(ser.readline().decode('utf-8'))
        print('serinput', serinput)
        # if motorOrTask == False: # wait until arduino finish task 
        #     if serinput == 999:
        #         motorOrTask = True
        #     else:
        #         break
        # if serdiff:
        #   if serinput == int(motorOutput[0:3]):
        #       serdiff = False
        #   else:
        #       ser.write(motorOutput.encode('utf-8'))
        #       print('there is serdiff')
        #       break
        # if serinput == 1:
        #   state = 1
        #   print('change state to 1')
        # elif serinput == 2:
        #   state = 3
        #   print('change state to 3')
        # elif serinput == 3:
        #   state = 5
        #   print('change state to 5')
        # elif serinput == 4:
        #   state = 9
        #   print('change state to 9')
        ret1, frontFrame = frontCap.read()
        # ret2, sideFrame = sideCap.read()
        # ret3, waterFrame = waterCap.read()
        if ret1:# and ret2 and ret3:
            # frontFrame = cv2.resize(frontFrame, (640, 480))
            # frontFrame = cv2.flip(frontFrame, -1)
            # sideFrame = cv2.resize(sideFrame, (640, 480))
            waterFrame = cv2.resize(waterFrame, (640, 480))
            
            # maskName['signShow'] = frontFrame.copy()
            # maskName['potShow'] = frontFrame.copy()
            # maskName['tubeShow'] = frontFrame.copy()
            maskAll()
            # switch()
            waterDetect()
            # tubeDetect()
            # signDetect()
        if motorOrTask:
            ser.write(motorOutput.encode('utf-8'))
            print('----ser motor----')
        else:
            ser.write(taskOutput.encode('utf-8'))
            print('----ser task----')
            print(taskOutput)
            motorOrTask = True
        if cv2.waitKey(1)==ord("q"):
            break
        break
    if cv2.waitKey(1)==ord("q"):
        break
        
# frontCap.release()
# sideCap.release()
waterCap.release()
cv2.destroyAllWindows()
