import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    #这里的XY代表要求的点，（x1，y1）（x2，y2）是用来确定直线用的
    a = lineY2-lineY1
    b = lineX1-lineX2
    c = lineX2*lineY1-lineX1*lineY2
    dis =np.abs((a*pointX+b*pointY+c)/(math.pow(a*a+b*b,0.5)))
    #注意：这里没有加绝对值，得出的数有正负之分
    print(a,b,c)
    #pow--根号下
    return dis

def fitlines(picture,threshold_result,points,):
    rows, cols = threshold_result.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(picture, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)

src = cv2.imread('blackline1.jpg')#读进来是RGB格式的
src = cv2.resize(src,(720,720))
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)#将图片转化为灰度图
ret,final_1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#找到轮廓
contours_final,hierarchy_final = cv2.findContours(final_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rows,cols = final_1.shape[:2]
[vx,vy,x,y] = cv2.fitLine(contours_final[4],cv2.DIST_L2,0,0.01,0.01)
#找出最优拟合直线
k = vy/vx #求出画处线的斜率
lefty = int((-x*vy/vx)+y)
righty = int(((cols-x)*vy/vx)+y)
cv2.drawContours(src,contours_final,4,(0,0,255),5)
cv2.line(src,(cols-1,righty),(0,lefty),(0,0,255),2)#画出最优拟合直线
contours_final_reshape = contours_final[4].reshape((1602,2))
distance = getDis(contours_final_reshape[:,0],contours_final_reshape[:,1],0,lefty,720,righty)
point_processing = contours_final_reshape[distance<80,:]#确定阈值
process = np.zeros(gray.shape,np.uint8)
process[point_processing[:,1],point_processing[:,0]] =255
line_process = cv2.line(process,(0,lefty),(720,righty),(255,0,0))

[vx2,vy2,x2,y2] = cv2.fitLine(point_processing,cv2.DIST_L2,0,0.01,0.01)
lefty2 = int((-x2*vy2/vx2)+y2)
righty2 = int(((cols-x2)*vy2/vx2)+y2)

cv2.line(src,(cols-1,righty2),(0,lefty2),(0,255,0),2)

cv2.imshow('process',process)
cv2.imshow('final',src)
cv2.waitKey()
cv2.destroyAllWindows()