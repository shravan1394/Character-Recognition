import cv2
import numpy as np
import scipy.io as sio
from oct2py import octave
from arrange import arrange
from matplotlib import pyplot as plt
mat0 = sio.loadmat('tr0.mat')
mat1 = sio.loadmat('tr1.mat')
mat2 = sio.loadmat('tr2.mat')
Theta0=mat0['Theta0']
Theta1=mat1['Theta1']
Theta2=mat2['Theta2']
cv2.destroyAllWindows() 
ind=np.arange(256).reshape(256,1)
himn=20
histd=0
def nothing(e):
	pass
m=0
k=0
std=0.0
O=[]
Q=0.0
yt=0
zt=0
#cv2.namedWindow('edges')
#cv2.namedWindow('edge')
#cv2.createTrackbar('h','edge',0,255,nothing)
#cv2.createTrackbar('s','edge',1,500,nothing)
#cv2.createTrackbar('v','edge',1,255,nothing)
#cv2.createTrackbar('h1','edges',0,255,nothing)
#cv2.createTrackbar('s1','edges',0,255,nothing)
#cv2.createTrackbar('v1','edges',0,255,nothing)
cv2.VideoCapture(0).release()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
while(cap.isOpened()):
	a=np.zeros(4,np.float32)
	b=np.zeros(4,np.float32)
	word=[]
	c=[]
	m=0
	
	ret, img = cap.read()
	_,img1=cap.read()
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv = cv2.GaussianBlur(img,(9,9),0)
	#cv2.imshow("aef",hsv)
	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	blur=cv2.GaussianBlur(img,(9,9),0)
	blur1=cv2.GaussianBlur(gray,(7,7),0)
	th2 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,16)
	#h = cv2.getTrackbarPos('h','edge')
	#s = cv2.getTrackbarPos('s','edge')
	#v = cv2.getTrackbarPos('v','edge')
	#h1 = cv2.getTrackbarPos('h1','edges')
	#s1 = cv2.getTrackbarPos('s1','edges')
	#v1 = cv2.getTrackbarPos('v1','edges')
	if cv2.waitKey(3) == ord('p'):
	       	cv2.imwrite("selfie.jpg",img)

	edges = cv2.Canny(blur,0,100,apertureSize = 3)
	edes = cv2.Canny(blur,0 ,100,apertureSize = 3)
	_,contours0, hierarchy0 = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	lower_blue = np.array([0,0,0],np.uint16)
    	upper_blue = np.array([180,104,255],np.uint16)
	mask = cv2.inRange(hsv,lower_blue, upper_blue)
	mas = cv2.inRange(hsv,lower_blue, upper_blue)
	_,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	lower_blue1 = np.array([0,0,0],np.uint16)
    	upper_blue1 = np.array([180,125,255],np.uint16)
	mask1 = cv2.inRange(hsv,lower_blue1, upper_blue1)
	mas1= cv2.inRange(hsv,lower_blue1, upper_blue1)
	_,contours1, hierarchy1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	lower_blue2 = np.array([0,0,0],np.uint16)
    	upper_blue2 = np.array([180,115,255],np.uint16)
	mask2 = cv2.inRange(hsv,lower_blue2, upper_blue2)
	mas2 = cv2.inRange(hsv,lower_blue2, upper_blue2)
	_,contours2, hierarchy2 = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	lower_blue3 = np.array([0,0,0],np.uint16)
    	upper_blue3 = np.array([255,84,255],np.uint16)
	mask3 = cv2.inRange(hsv,lower_blue3, upper_blue3)
	mas3 = cv2.inRange(hsv,lower_blue3, upper_blue3)
	_,contours3, hierarchy3 = cv2.findContours(mask3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	hist_full0 = cv2.calcHist([blur1],[0],None,[256],[0,256]);
	plt.plot(hist_full0);
	
	#print himn
	_,tt = cv2.threshold(blur1,himn-2*histd,255,cv2.THRESH_BINARY_INV)
	#_,ttt = cv2.threshold(blur1,h,255,cv2.THRESH_BINARY_INV)
	#cv2.imshow("jf",ttt)
	_,contours4, hierarchy4 = cv2.findContours(tt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for i in range(len(contours1)):
		cnt=contours1[i]
		epsilon = 0.01*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print cv2.contourArea(cnt)
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)
			m+=1

	for i in range(len(contours2)):
		cnt=contours2[i]	
		epsilon = 0.01*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print cv2.contourArea(cnt)
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)
			m+=1

	for i in range(len(contours3)):
		cnt=contours3[i]	
		epsilon = 0.01*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print cv2.contourArea(cnt)
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)
			m+=1

	for i in range(len(contours0)):
		cnt=contours0[i]	
		epsilon = 0.1*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print approx
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)	
			m+=1	
	for i in range(len(contours)):
		cnt=contours[i]	
		epsilon = 0.01*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print cv2.contourArea(cnt)
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)	
			m+=1	
	for i in range(len(contours4)):
		cnt=contours4[i]	
		epsilon = 0.01*cv2.arcLength(cnt,True)
    		approx = cv2.approxPolyDP(cnt,epsilon,True)
		if len(approx)==4 and cv2.contourArea(cnt)>100000 and cv2.contourArea(cnt)<250000: 
			#print cv2.contourArea(cnt)
			pts1=np.float32(approx.reshape(4,2))
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(img,[cnt],0,(0,255,0),1)	
			m+=1	
	
	if m>0:
		#im0=img[y:(y+h),x:(x+w)]
		l=pts1[:,1]**2+pts1[:,0]**2
		l=l.reshape(4,1)
		a[0]=pts1[np.where(np.any(l==min(l),axis=1)),0]+20
		b[0]=pts1[np.where(np.any(l==min(l),axis=1)),1]+20
		a[3]=pts1[np.where(np.any(l==max(l),axis=1)),0]-20
		b[3]=pts1[np.where(np.any(l==max(l),axis=1)),1]-20
		a[1]=pts1[np.where(np.any((l!=max(l)) & (l!=min(l)) & (pts1[:,0]<pts1[:,1]).reshape(4,1),axis=1)),0]+20
		b[1]=pts1[np.where(np.any((l!=max(l)) & (l!=min(l)) & (pts1[:,0]<pts1[:,1]).reshape(4,1),axis=1)),1]-20
		a[2]=pts1[np.where(np.any((l!=max(l)) & (l!=min(l)) & (pts1[:,0]>pts1[:,1]).reshape(4,1),axis=1)),0]-20
		b[2]=pts1[np.where(np.any((l!=max(l)) & (l!=min(l)) & (pts1[:,0]>pts1[:,1]).reshape(4,1),axis=1)),1]+20
		pts1 = np.float32([[a[0],b[0]],[a[1],b[1]],[a[2],b[2]],[a[3],b[3]]])
    		pts2 = np.float32([[0,0],[0,300],[450,0],[450,300]])
		M = cv2.getPerspectiveTransform(pts1,pts2)

    		d = cv2.warpPerspective(img,M,(450,300))
		ds = cv2.warpPerspective(gray,M,(450,300))
		dst = cv2.warpPerspective(th2,M,(450,300))
		dst1 = cv2.warpPerspective(th2,M,(450,300))
		hist_full = cv2.calcHist([ds],[0],None,[256],[0,256])
		himn=np.average(ind,weights=hist_full)
		#print np.average(ind,weights=hist_full)
		histd=(np.average((ind-np.ones((256,1))*himn)**2,weights=hist_full))**0.5
		
		ret,t = cv2.threshold(ds,himn-2*histd,255,cv2.THRESH_BINARY_INV)
		ret,td = cv2.threshold(ds,himn-2*histd,255,cv2.THRESH_BINARY_INV)
		_,contous, hierarch = cv2.findContours(td,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)			
        	#cv2.imshow("asd",t)
		#cv2.imshow("ad",td)
		if cv2.waitKey(3) == ord('p'):
	
        		cv2.imwrite("lettrs.png",d)
		(cont,indi,indl)=arrange(contous,hierarch)
		for i in range(len(cont)):
			cn=cont[i]
			x1,y1,w1,h1 = cv2.boundingRect(cn)
			cv2.rectangle(d,(x1,y1),(x1+w1,y1+h1),(0,255,0),1)
			im0=t[y1:(y1+h1),x1:(x1+w1)]
			black=np.zeros((np.shape(t)))
			
			#kernel = np.ones((3,3),np.uint8)
			#im0 = cv2.erode(im0,kernel,iterations = 1)
			black[y1:(y1+h1),x1:(x1+w1)]=im0
			im0=black[y1-h1/5:(y1+h1+h1/5),x1-w1/3:(x1+w1+w1/3)]
			if w1/float(h1)<0.3:
				im0=black[y1-h1/5:(y1+h1+h1/5),x1-3*w1:(x1+w1+3*w1)]
			res = cv2.resize(im0,(20, 20), interpolation = cv2.INTER_CUBIC)
			#print (w1/float(h1))
			#cv2.imshow('edge',res)
			#cv2.imshow('edge',d)
			#cv2.waitKey(0)
						
			res=res.astype(float)
			cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
			#print res
			l=np.transpose(res).reshape(1,400)
			c.append(l)
			l=np.array(l)
			#print np.shape(l)
			#p=octave.predict(Theta1,Theta2,l)
			#print chr(int(p)+64)				
			#k=k+1
		#cv2.imshow("ex",d)
		c=np.array(c)
		u,o,r=np.shape(c)
		#sio.savemat('RMI.mat', {'vect':c})
		#break;
		#cv2.imshow('edge',t)
	#cv2.imshow('ed',mas1)
		#print np.shape(c)
		c=c.reshape(u,r)
		p=octave.predict(Theta0,Theta1,Theta2,c);
		#print p
		#for i in range(len(p)):
		#	word.append(chr(p[i]+64));	
		#print "".join(word)
		if k<8 and k>=3:
			Q+=np.size(p)
			std+=np.size(p)**2
			
			#print np.size(p)
		if k==8:
			#print std
			#print Q
			std=((std/5)-(Q/5)**2)**0.5
			Q=np.round(Q/5)
			Q=int(Q)
			#print std
			#print Q
			O=p[0:Q+1]
			if std>0.5:
				print 1
				break
		elif k>8 and np.size(p)==Q:
			
			#print O
			#print O
			for i in range(len(p)):
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
				word.append(chr(p[i]+64));	
			
			print "".join(word)
			break
			
			#break
		#cv2.destroyAllWindows()
		#break
		k=k+1
	#cv2.imshow('e',mas2)
	cv2.imshow("edge",img1)
	#print k
	
	#cv2.imshow('ex',edes)
	#cv2.waitKey(0)
	#plt.hist(img.ravel(),256,[0,256]); plt.show()
	if cv2.waitKey(3)==ord('q'):
		break
cap.release() 
cv2.waitKey(3)
cv2.destroyAllWindows()

   

