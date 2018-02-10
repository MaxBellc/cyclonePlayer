#!/usr/bin/env python






# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

# relative module
import video

# built-in module
import sys


if __name__ == '__main__':

	try:
		fn = sys.argv[1]
	except:
		fn = 0
	def nothing(*arg):
		pass


	cap = video.create_capture(fn)


	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image',600,600)
	thrs=50
	cv2.createTrackbar('r', 'image', 81, 255-thrs, nothing)
	cv2.createTrackbar('g', 'image', 57, 255-thrs, nothing)
	cv2.createTrackbar('b', 'image', 39, 255-thrs, nothing)
	cv2.createTrackbar('thresh', 'image', 8, 100, nothing) 

   #sets how much to blur
	filt=11

	while True:
		flag, img = cap.read()
		
	  
		r = cv2.getTrackbarPos('r', 'image')
		g = cv2.getTrackbarPos('g', 'image')
		b = cv2.getTrackbarPos('b', 'image')
		threshol = cv2.getTrackbarPos('thresh', 'image')
		

		lower=[b,g,r]
		lower=np.array(lower, dtype="uint8")
	   
		
		mask=cv2.inRange(img,lower,lower+thrs)
	 
		vis = img.copy()
		vis = np.uint8(vis)
		mask=np.uint8(mask)
		vis[mask==0]=(0,0,0)
		
		gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (filt, filt), 0)
		thresh = cv2.threshold(blurred, threshol, 255, cv2.THRESH_BINARY)[1]



		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[1]

		areas=len(cnts)
		areas=int(areas)
		#print(areas)
		splotch = np.zeros((1,areas),dtype=np.uint8)
		i=-1
		#print(splotch)
		
		# loop over the contours
		for c in cnts:
		
			i=i+1
			M = cv2.moments(c)
			splotch[0][i] = int(M["m00"])
		try:
			max1=np.argmax(splotch)
		except:
			max1=-1
		print(max1)
		original=vis.copy()
		
		if max1>-1:
			M = cv2.moments(cnts[max1])
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])


			
			cv2.drawContours(vis, [cnts[max1]], -1, (0, 255, 0), 2)
			cv2.circle(vis, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(vis, "center", (cX - 20, cY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



		
		
		cv2.imshow('image',np.hstack([thresh, gray2])) #np.hstack([original, vis]))#np.hstack([thresh, gray2]))
		ch=cv2.waitKey(5)
		if ch == 27:
			break

	cv2.destroyAllWindows()
















        
    
