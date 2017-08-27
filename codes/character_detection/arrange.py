#!/usr/bin/python
# arranges the contours based on thiers proximity to top left corner of the page
def arrange(contours,hierarchy):
	import cv2
	import numpy as np
	j=0
	cx=[]
	cy=[]
	sx=[]
	sy=[]
	indi=[]
	indl=[]
	s=[]
	d=[]
	a=[]
	f=[]
	con=[]
	cont=[]
	for i in range(len(contours)):
				cnt=contours[i]
				hie=hierarchy[0,i]
				#if cv2.contourArea(cnt)>193:
				#print cv2.contourArea(cnt)
				x,y,w,h = cv2.boundingRect(cnt)
				
				if hie[3]==-1 and w*h>100: 
					con.append(cnt)
					M = cv2.moments(cnt)
					cx.append(M['m10']/M['m00'])
					cy.append(M['m01']/M['m00'])
	f=[i for i in range(0,len(con))]
	#print (f)	
	c=np.vstack((cx,cy,f)).T
	c=c[c[:,1].argsort()]
	#print (c)
	cy.sort()
	
		
	for i in range(len(cy)-1):
		#print (cx[i+1]-cx[i])/(area[i+1]-area[i])
		s.append(cy[i+1]-cy[i])
		#print s

		#print area
	for i in range(len(s)):
		#print s[i]
		if s[i]>np.mean(s)+2.5*np.std(s):
			
			d=c[j:i+1,:]
			d=d[d[:,0].argsort()]
			a.append(d)
			#print d
			j=i+1
		
		if i==len(s)-1:
			d=c[j:,:]
			d=d[d[:,0].argsort()]
			a.append(d)
	
	a=np.vstack(a)
	#print a
	for i in range(len(con)):
		cont.append(con[int(a[i,2])])
	#print np.shape(cont)
	for i in range(len(a[:,0])-1):
		sy.append(a[i+1,1]-a[i,1])
		sx.append(a[i+1,0]-a[i,0])
	for i in range(len(sx)):
		if sx[i]>np.mean(sx)+1.3*np.std(sx):
			indi.append(i)
		if sx[i]<0 or sx[i]>np.mean(sx)+2.5*np.std(sx) or sy[i]>np.mean(sy)+2.5*np.std(sy):
			indl.append(i)
	return (cont,indi,indl)
	
