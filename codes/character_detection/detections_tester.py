#!/usr/bin/python

# detection testing code for images
from arrange import arrange
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from oct2py import octave
import cv2
word=[]
ma=1
mat0 = sio.loadmat('tr0.mat')
mat1 = sio.loadmat('tr1.mat')
mat2 = sio.loadmat('tr2.mat')
Theta0=mat0['Theta0']
Theta1=mat1['Theta1']
Theta2=mat2['Theta2']
 
def nothing(x):
	pass
cv2.namedWindow('edges')
cv2.createTrackbar('ma', 'edges',90,500,nothing)
cv2.createTrackbar('mi', 'edges',30,100,nothing)
cv2.createTrackbar('th', 'edges',1,255,nothing)
X=np.zeros((6,400))

img=cv2.imread("2.jpg")
ma=1
f=1
k=0
ind=np.arange(256).reshape(256,1)
yt=0
zt=0
while 1:
	img=cv2.imread("chk1.jpg")
	ma = cv2.getTrackbarPos('ma','edges')
	mi = cv2.getTrackbarPos('mi','edges')
	th = cv2.getTrackbarPos('th','edges')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	#ma = cv2.getTrackbarPos('ma','edges')
	#mi = cv2.getTrackbarPos('mi','edges')
	#th = cv2.getTrackbarPos('th','edges')
	th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,16)
	#plt.hist(gray.ravel(),256,[0,256]); plt.show()
	hist_full = cv2.calcHist([gray],[0],None,[256],[0,256])
	#hist_mask = cv2.calcHist([gray],[0],mask,[256],[0,256])
	#plt.plot(hist_full);plt.show()
	#print np.shape(hist_full)
	edges = cv2.Canny(blur,mi,ma,apertureSize = 3)
	himn=np.average(ind,weights=hist_full)
	histd=(np.average((ind-np.ones((256,1))*himn)**2,weights=hist_full))**0.5
	#print (np.average(ind,weights=hist_full))
	#print(np.average((ind-np.ones((256,1))*himn)**2,weights=hist_full))**0.5
	#print (np.argmax(hist_full))
	ret,t = cv2.threshold(gray,himn-1.3*histd,255,cv2.THRESH_BINARY_INV)
	#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	ret,td = cv2.threshold(gray,himn-1.3*histd,255,cv2.THRESH_BINARY_INV)
         #   cv2.THRESH_BINARY_INV,(11),th)
	
	_,contours, hierarchy = cv2.findContours(td,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#contours1, hierarchy1 = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#print hierarchy
	
				
				
	(cont,indi,indl)=arrange(contours,hierarchy)
	
	for i in range(len(cont)):
			cnt=cont[i]
			
		#if cv2.contourArea(cnt)>193:
		#print cv2.contourArea(cnt)
			x,y,w,h = cv2.boundingRect(cnt)
			
			#print x
			#print y
			#print w
			#print h 
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
			im0=t[y:(y+h),x:(x+w)]
			black=np.zeros((np.shape(t)))
			
			#kernel = np.ones((3,3),np.uint8)
			#im0 = cv2.erode(im0,kernel,iterations = 1)
			black[y:(y+h),x:(x+w)]=im0
			im0=black[y-h/5:(y+h+h/5),x-w/3:(x+w+w/3)]
			M = cv2.moments(cnt)
			cx = (M['m10']/M['m00'])
			cy = (M['m01']/M['m00'])
			
			if w/float(h)<0.4:
				im0=black[y-h/5:(y+h+h/5),x-3*w:(x+w+3*w)]
			res = cv2.resize(im0,(20, 20), interpolation = cv2.INTER_CUBIC)
			#cv2.imshow('edge',im0)
			#cv2.imshow('edgeswx',res)
			#cv2.imshow('edges1',img)
			#cv2.waitKey(0)
			
			res=res.astype(float)
			cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
			#print res
			if indi:
				if i==indi[yt]+1:
					word.append(" ")
					if yt< len(indi)-1:
							yt=yt+1
			if indl:
				if i==indl[zt]+1:
					word.append(" ")
					if zt< len(indl)-1:
							zt=zt+1
			l=np.transpose(res).reshape(1,400)
			#l=octave.change(l);
			p=octave.predict(Theta0,Theta1,Theta2,l)
			word.append( chr(int(p)+64))	
			#X[k,:]=l
			#print k				
			k=k+1
	#print "".join(word)			
	#sio.savemat('R.mat', {'vect':X})
	cv2.imshow('edges',edges)
	#cv2.imshow('edges1',img)
	#print k
	#cv2.destroyAllWindows()
	#cv2.waitKey(0)	
	#break;
	
	
	
	
	
	

	
	
	
	if cv2.waitKey(3)==ord('q'):
		break
    


  
    
    
     

cv2.destroyAllWindows()

