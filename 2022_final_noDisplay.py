import cv2
import numpy as np
import serial
import time

ser = serial.Serial('COM17', 9600)
motorOutput = "000000\n"
taskOutput = "900000\n"
motorOrTask = True 	#decide to send which output to arduino (true motor/ false task)
variables = dict.fromkeys(['signCode', 'signCounter', 'colorCode', 'colorCounter'], 0)
pixelcount = []
Ilist = []
prevtime = time.time()
setprevtime = 0
Iterm = 0
fruitName = ['tomato', 'lemon', '', 'avocado']
FruitColorHSV=[ [  0,113,  0, 12,255,200], [19, 130, 131, 30, 255, 255], [], [25, 22, 8, 94, 255, 98]]#red, yellow, none, black
for i in range(640):
	pixelcount.append(0)

#						sign 			pot 			tube 		   red 			yellow 			blue 			black
maskLowwerBound = [[  0,128,198], [  0,117,  0], [  0, 42,136], [  0,174,146], [  0,138,190], [ 74, 85,  0], [ 31,  0, 14]]
maskUpperBound  = [[ 19,253,255], [  9,223,186], [179,255,255], [179,255,255], [ 41,255,255], [108,255,255], [ 71,255,111]]
maskName = dict.fromkeys(['signMask', 'potMask', 'tubeMask', 'redSideMask', 'yellowSideMask', 'blueSideMask', 'blackSideMask', 'redWaterMask', 'yellowWaterMask', 'blueWaterMask', 'blackWaterMask', 'potShow', 'signShow', 'tubeShow', 'fruitShow']) 
state = 1

def maskAll():	#process masks	input: three caps/ output: eleven masked img (sign, pot, tube, 4 colors)
	frontHsv=cv2.cvtColor(frontFrame,cv2.COLOR_BGR2HSV)
	sideHsv=cv2.cvtColor(sideFrame,cv2.COLOR_BGR2HSV)
	waterHsv=cv2.cvtColor(waterFrame,cv2.COLOR_BGR2HSV)
	for i in range(3):	#mask sign, pot, tube
		mask = cv2.inRange(frontHsv,np.array(maskLowwerBound[i]),np.array(maskUpperBound[i]))
		maskName[list(maskName)[i]] = cv2.bitwise_and(frontFrame,frontFrame,mask=mask)
	maskName['signMask'] = cv2.cvtColor(maskName['signMask'],cv2.COLOR_BGR2GRAY)
	for i in range(3, 7):	#mask side four color
		mask = cv2.inRange(sideHsv,np.array(maskLowwerBound[i]),np.array(maskUpperBound[i]))
		maskName[list(maskName)[i]] = cv2.bitwise_and(sideHsv,sideHsv,mask=mask)
	for i in range(7, 11):	#mask water four color
		mask = cv2.inRange(waterHsv,np.array(maskLowwerBound[i-4]),np.array(maskUpperBound[i-4]))
		maskName[list(maskName)[i]] = cv2.bitwise_and(waterFrame, waterFrame,mask=mask)
		maskName[list(maskName)[i]] = cv2.cvtColor(maskName[list(maskName)[i]],cv2.COLOR_BGR2GRAY)

def signDetect():	#input: mask sign img/ output: (int)variables['signCode'] (0: None, 1: left tri, 2: right tri, 3: square, 4: circle) and return True
	contours, hierarchy = cv2.findContours(maskName['signMask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
	# x,y,w,h=-1,-1,-1,-1
	for cnt in contours:
		area = cv2.contourArea(cnt)  
		# cv2.drawContours(maskName['signShow'], cnt, -1, (0, 255, 0), 3)
		if area > 15000:  
			peri = cv2.arcLength(cnt, True)  
			vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)  
			corners = len(vertices)  
			x, y, w, h = cv2.boundingRect(vertices)  
			# cv2.putText(maskName['signShow'], str(area), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			if corners==3:
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
					# cv2.putText(maskName['signShow'], "left", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					variables['signCode'] = '1'
					# cv2.imshow('sign', maskName['signShow'])
					return True
				elif flag_x <= sec_x:
					# cv2.putText(maskName['signShow'], "right", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					variables['signCode'] = '2'
					# cv2.imshow('sign', maskName['signShow'])
					return True
			elif corners == 4: 
				# cv2.putText(maskName['signShow'], "rectangle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				variables['signCode'] = '3'
				# cv2.imshow('sign', maskName['signShow'])
				return True
			elif corners == 8:  
				# cv2.putText(maskName['signShow'], "circle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				variables['signCode'] = '4'
				# cv2.imshow('sign', maskName['signShow'])
				return True
	# cv2.putText(maskName['signShow'], "None", (10, 480-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	# cv2.imshow('sign', maskName['signShow'])
	return False

def potDetect():	#input: mask pot img/ output: (change global variable) motorOutput
	global motorOutput
	global taskOutput
	turnratio = 0.3
	firstcoor = -1
	upbound = 250
	lowbound = 330
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
			# maskName['potShow'] = cv2.line(maskName['potShow'], (i, 480), (i, 480-pixelcount[i]), (255, 255, 0), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[0], 480), (boundary[0], 0), (0, 0, 255), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[1], 480), (boundary[1], 0), (0, 0, 255), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[2], 480), (boundary[2], 0), (0, 0, 255), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (boundary[3], 480), (boundary[3], 0), (0, 0, 255), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (0, upbound), (640, upbound), (0, 0, 255), 1)
			# maskName['potShow'] = cv2.line(maskName['potShow'], (0, lowbound), (640, lowbound), (0, 0, 255), 1)
	if abs(leftdiff-rightdiff) < 25:
		# cv2.putText(maskName['potShow'], "go straight", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
	elif leftdiff > rightdiff:
		# cv2.putText(maskName['potShow'], "turn right", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
	else:
		# cv2.putText(maskName['potShow'], "turn left", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
	left = 128 + moveratio 
	right = 128 - moveratio
	left_str = str(int(left))
	right_str = str(int(right))
	if right < 100:
		right_str = '0' + right_str
	if left < 100:
		left_str = '0' + left_str
	motorOutput = left_str + right_str +'\n'
	# print('boundary:', boundary)
	# print('leftdiff:', leftdiff, 'rightdiff:', rightdiff)
	# print(motorOutput)
	# cv2.imshow('pot', maskName['potShow'])
	# cv2.imshow('pot', maskName['potShow'])

def colorDetect():	#input: four color mask img/ output: (change global variable) colorSign and return True
	for i in range(4):
		maskName[list(maskName)[i+3]] = cv2.cvtColor(maskName[list(maskName)[i+3]],cv2.COLOR_HSV2BGR)
		maskName[list(maskName)[i+3]] = cv2.cvtColor(maskName[list(maskName)[i+3]],cv2.COLOR_BGR2GRAY)
		contours, hierarchy = cv2.findContours(maskName[list(maskName)[i+3]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			if cv2.contourArea(cnt) > 7000: 
				if state ==  1:
					variables['colorCode'] = i
					return True
				elif state == 5:
					variables['colorCode'] = i+4
					return True
	return False
	# maskName[list(maskName)[variables['colorCode']]] = cv2.cvtColor(maskName[list(maskName)[variables['colorCode']]],cv2.COLOR_GRAY2BGR)

def findFruitContour(img,i):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
	for cnt in contours:
		# cv2.drawContours(maskName['fruitShow'], cnt, -1, (0, 255, 0), 3)
		area = cv2.contourArea(cnt)  
		if area >25000:  
			# x, y, w, h = cv2.boundingRect(cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True))  
			# cv2.putText(maskName['fruitShow'], str(area), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			# cv2.putText(maskName['fruitShow'], fruitName[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			return True
	return False

def fruitDetect():
	global motorOutput, taskOutput, motorOrTask
	fruitFrame = waterFrame[50:420, 0:420]
	maskName['fruitShow'] = fruitFrame.copy()
	hsv=cv2.cvtColor(fruitFrame,cv2.COLOR_BGR2HSV)
	i = variables['colorCode']
	lower=np.array(FruitColorHSV[i][:3])
	upper=np.array(FruitColorHSV[i][3:6])
	mask=cv2.inRange(hsv,lower,upper)
	result = cv2.bitwise_and(fruitFrame,fruitFrame,mask=mask)
	while findFruitContour(mask,i) == False:
		motorOutput = "032032\n"
		# cv2.imshow("result", result)
		# cv2.imshow("contour", maskName['fruitShow'])
		return False
	taskOutput = "900010\n"
	motorOrTask = False
	# cv2.imshow("result", result)
	# cv2.imshow("contour", maskName['fruitShow'])	
	return True

def waterDetect():	#input: four color mask img/ output: loop until water
	contours, hierarchy = cv2.findContours(maskName[list(maskName)[variables['colorCode']+7]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	global motorOutput
	global taskOutput
	global motorOrTask
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 7000:
			# cv2.drawContours(waterFrame, cnt, -1, (128, 0, 128), 3)
			x, y, w, h = cv2.boundingRect(cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True))
			cv2.circle(waterFrame, (int(x+w/2), int(y+h/2)), 7, (255, 255, 255), -1)
			if x+w/2 < 300:
				# cv2.putText(waterFrame, "motor forward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
				motorOutput = '030030\n'
			elif x+w/2 > 340:
				# cv2.putText(waterFrame, "motor backward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
				motorOutput = '850850\n'
			else:
				if y+h/2 < 300:
					# cv2.putText(waterFrame, "stepper forward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
					taskOutput = '902000\n'
					motorOrTask = False
				elif y+h/2 > 340:
					# cv2.putText(waterFrame, "stepper backward", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
					taskOutput = '901000\n'
					motorOrTask = False
				else:
					# cv2.putText(waterFrame, "water!!", (10, 480-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (128, 0, 128), 2)
					taskOutput = '900100\n'
					motorOrTask = False
					return True
		# cv2.rectangle(waterFrame, (300,220), (340, 260), (255, 255, 255), 3)
	# cv2.imshow('result', waterFrame)
	return False

def tubeDetect():	#input: mask tube img/ output: (change global variable) motorOutput
	global motorOutput
	global tubeQue
	global Iterm
	global prevtime
	sideratio, turnratio, luCounter, llCounter ,ruCounter, rlCounter = 0.1, 1.2, 0, 0, 0, 0
	upbound = 100
	lowbound = 380
	midpoint = 240
	setpoint = 100
	boundary = [0, 0, 0]
	num = np.uint8(0)
	for i in range(0, 480):
		#chop the image
		if boundary[0] == 0:
			if (maskName['tubeMask'][upbound][i][0] != num or maskName['tubeMask'][upbound][i][1] != num or maskName['tubeMask'][upbound][i][2] != num):
				if luCounter > 2:
					boundary[0] = i
					# cv2.circle(maskName['tubeShow'], (i, upbound), 5, (255, 204, 0), -1)
				else :
					luCounter += 1
			else:
				luCounter = 0
		if boundary[1] == 0:
			if (maskName['tubeMask'][lowbound][i][0] != num or maskName['tubeMask'][lowbound][i][1] != num or maskName['tubeMask'][lowbound][i][2] != num):
				if llCounter > 2:
					boundary[1] = i
					# cv2.circle(maskName['tubeShow'], (i, lowbound), 5, (255, 204, 0), -1)
				else :
					llCounter += 1
			else:
				llCounter = 0
		if boundary[2] == 0:
			if (maskName['tubeMask'][midpoint][i][0] != num or maskName['tubeMask'][midpoint][i][1] != num or maskName['tubeMask'][midpoint][i][2] != num):
				if ruCounter > 2:
					boundary[2] = i
					# cv2.circle(maskName['tubeShow'], (i, midpoint), 5, (255, 204, 0), -1)
				else :
					ruCounter += 1
			else:
				ruCounter = 0

		
	sideratio = 0.4
	leftdiff = boundary[0]
	rightdiff = boundary[1]
	middiff = boundary[2] - setpoint
	moveratio = (leftdiff-rightdiff) * turnratio - middiff * sideratio
	# cv2.line(maskName['tubeShow'], (setpoint, 0), (setpoint, 480), (0, 255, 0), 1)
	left = 128 + moveratio 
	right = 128 - moveratio
	left_str = str(int(left))
	right_str = str(int(right))
	if right < 100:
		right_str = '0' + right_str
	if left < 100:
		left_str = '0' + left_str
	motorOutput = left_str + right_str +'\n'
	# tubeQue.append(now)
	# motorOutput = tubeQue.pop(1)
	# cv2.imshow('tube', maskName['tubeShow'])
	# print('boundary:', boundary)
	# print('leftdiff:', leftdiff, 'rightdiff:', rightdiff)
	# print('now',now)
	# print('motorOutput', motorOutput)

def potAndSign(currentState):
	global state
	global taskOutput
	global motorOrTask
	potDetect()
	if signDetect():
		# cv2.imshow('sign', maskName['signShow'])
		if variables['signCounter'] > 3:		# counter to be tested
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
	global state, motorOrTask, taskOutput
	potDetect()
	if colorDetect():
		if variables['colorCounter'] > 3:
			state = 1 + currentState
			variables['colorCounter'] = 0
			taskOutput = '90000'+str(variables['colorCode']+1)+'\n'
			motorOrTask = False
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
		if fruitDetect():
			state = 3
	elif state == 3:
		potAndSign(state)
	elif state == 4:
		potAndSign(state)
	elif state == 5:
		potAndColor(state)
	elif state == 6:
		if waterDetect():
			state = 7
	elif state == 7:
		potAndSign(state)
	elif state == 8:
		potAndSign(state)
	elif state == 9:
		tubeDetect()


frontCap = cv2.VideoCapture(0)
sideCap = cv2.VideoCapture(2)
waterCap = cv2.VideoCapture(1)
ser.write(motorOutput.encode('utf-8'))
while True:
	while ser.in_waiting:
		serinput = int(ser.readline().decode('utf-8'))
		# print('serinput', serinput)
		if motorOrTask == False: # wait until arduino finish task 
			if serinput == 999:
				motorOrTask = True
			else:
				# print('skip')
				break
		if serinput == 1:
			state = 1
			# print('change state to 1')
		elif serinput == 2:
			state = 3
			# print('change state to 3')
		elif serinput == 3:
			state = 5
			# print('change state to 5')
		elif serinput == 4:
			state = 9
			# print('change state to 9')
		ret1, frontFrame = frontCap.read()
		ret2, sideFrame = sideCap.read()
		ret3, waterFrame = waterCap.read()
		if ret1 and ret2 and ret3:
			frontFrame = cv2.resize(frontFrame, (640, 480))
			frontFrame = cv2.flip(frontFrame, -1)
			sideFrame = cv2.resize(sideFrame, (640, 480))
			waterFrame = cv2.rotate(waterFrame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			waterFrame = cv2.resize(waterFrame, (640, 480))
			maskName['signShow'] = frontFrame.copy()
			maskName['potShow'] = frontFrame.copy()
			maskName['tubeShow'] = frontFrame.copy()
			maskAll()
			switch()
			# fruitDetect()
			# cv2.imshow('front', frontFrame)
			# cv2.imshow('side', sideFrame)
			# cv2.imshow('water', waterFrame)
		if motorOrTask:
			ser.write(motorOutput.encode('utf-8'))
			# print('----ser motor----')
		else:
			ser.write(taskOutput.encode('utf-8'))
			# print('----ser task----')
			print(taskOutput)
		if cv2.waitKey(1)==ord("q"):
			break
		break
	if cv2.waitKey(1)==ord("q"):
		break
		
frontCap.release()
sideCap.release()
waterCap.release()
cv2.destroyAllWindows()
