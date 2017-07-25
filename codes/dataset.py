#!/usr/bin/python
import scipy.io as sio
import numpy as np
import cv2
ma=1
def nothing(x):
	pass
cv2.namedWindow('edges')
cv2.createTrackbar('ma', 'edges',0,10000,nothing)
cv2.createTrackbar('mi', 'edges',0,10000,nothing)
cv2.createTrackbar('th', 'edges',1,255,nothing)
data=[]
value=[]
img=cv2.imread("4.jpg")
ma=1
m=0
f=1
k=1
g=1
e=1
while 1:
	if g<10:
		value.append([g+47])
		if e<10:
			img=cv2.imread("Fnt/Sample00"+str(g)+"/img00"+str(g)+"-0000"+str(e)+".png")
		elif e>=10 and e<100:
			img=cv2.imread("Fnt/Sample00"+str(g)+"/img00"+str(g)+"-000"+str(e)+".png")
		elif e>=100 and e<1000:
			img=cv2.imread("Fnt/Sample00"+str(g)+"/img00"+str(g)+"-00"+str(e)+".png")
		else:
			img=cv2.imread("Fnt/Sample00"+str(g)+"/img00"+str(g)+"-0"+str(e)+".png")
	else:
		if g==10:
			value.append([g+47])
		elif g>10 and g<37:
			value.append([g+54])
		else:
			value.append([g+60])
		if e<10:
			img=cv2.imread("Fnt/Sample0"+str(g)+"/img0"+str(g)+"-0000"+str(e)+".png")
		elif e>=10 and e<100:
			img=cv2.imread("Fnt/Sample0"+str(g)+"/img0"+str(g)+"-000"+str(e)+".png")
		elif e>=100 and e<1000:
			img=cv2.imread("Fnt/Sample0"+str(g)+"/img0"+str(g)+"-00"+str(e)+".png")
		else:
			img=cv2.imread("Fnt/Sample0"+str(g)+"/img0"+str(g)+"-0"+str(e)+".png")
	#ma = cv2.getTrackbarPos('ma','edges')
	#mi = cv2.getTrackbarPos('mi','edges')
	#th = cv2.getTrackbarPos('th','edges')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	#ma = cv2.getTrackbarPos('ma','edges')
	#mi = cv2.getTrackbarPos('mi','edges')
	#th = cv2.getTrackbarPos('th','edges')
	th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,16)
	ret,t = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         #   cv2.THRESH_BINARY_INV,(11),th)
	
	res = cv2.resize(t,(20, 20), interpolation = cv2.INTER_CUBIC)
	res1=res.astype(float)
	cv2.normalize(res1, res1, 0, 1, cv2.NORM_MINMAX)
	#print res
	l=np.transpose(res1).reshape(1,400)
	data.append(l)
	#print k				
	k=k+1
	
	#cv2.imshow('edges',t)
	#cv2.imshow('edges1',img)
	#cv2.imshow('edge',res)
	#cv2.waitKey(0)
	print e
	print g
	e=e+1
	if e==1017:
		#print k
		k=1
		g+=1
		e=1
				
	if g==63:
		data=np.array(data)	
		value=np.array(value)		
		sio.savemat('data.mat', {'X':data,'y':value})
		break;
		
	
	
	
	
	

	
	
	
	if cv2.waitKey(3)==ord('q'):
		break
    



  
    
    
     

cv2.destroyAllWindows()

