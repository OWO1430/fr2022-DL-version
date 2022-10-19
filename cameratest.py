import cv2
import numpy as np

frontCap = cv2.VideoCapture(0)

while True:
	ret1, frontFrame = frontCap.read()
	if ret1:
		frontFrame = cv2.resize(frontFrame, (640, 480))
		cv2.imshow('front', frontFrame)
	else:
		print('error')
	if cv2.waitKey(1)==ord("q"):
		break
frontCap.release()
cv2.destroyAllWindows()
