# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:22:05 2018

@author: cchappey
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from PIL import Image as im
from resizeimage import resizeimage




###################################################Begin Calibration###################################"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
#objp=objp.reshape(-1,1,3)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('left_*.jpg')

img=cv2.imread("left_01.jpg")
img=im.fromarray(img)

#img.show()
img = img.resize([352,480], im.ANTIALIAS)
img.show()
img=np.array(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

for fname in images:
    print(images)
    img = cv2.imread(fname)
    img=im.fromarray(img)
    img = img.resize([352,480], im.ANTIALIAS)
    img=np.array(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
    print(np.size(corners))
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (5,5), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread('left_09.jpg')
img=im.fromarray(img)
img = img.resize([352,480], im.ANTIALIAS)
img=np.array(img)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))





# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
#l'image sortie est recalibre et tres sympathique

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ("total error: ", mean_error/len(objpoints))


###########################################POSE ESTIMATION################################################
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

##Creation critere de fin; et création des axes

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((5*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


###glob permet de tirer toutes les images 01...15
#Calcul de la rotation et de la translation en utilisant l'algorithme de Ransac
#Une fois les matrices de transformations trouvés ont on les projette sur les axes de l'image plane
#On retrouve les points du plan image correspondant à l'espace 3D
#Une fois ces points trouvés on cree les lignes du premier corner à chaqun de spoints trouvés

for fname in glob.glob('left_*.jpg'):
    img = cv2.imread(fname)
    img=im.fromarray(img)
    img = img.resize([352,480], im.ANTIALIAS)
    img=np.array(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()

###Creattion des cubes unité

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img
    
    
    
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
                   
                   
                   
for fname in glob.glob('left_*.jpg'):
    img = cv2.imread(fname)
    img=im.fromarray(img)
    img = img.resize([352,480], im.ANTIALIAS)
    img=np.array(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()




#####################################################################################

###############################Debut Repres 3d####################################"""
####################image 1#############################################"

depth_map=cv2.imread('dispartitytest.png')
depth_map = cv2.cvtColor(depth_map,cv2.COLOR_BGR2GRAY)
depth_map=np.array(depth_map)
height=np.size(depth_map,0)
width=np.size(depth_map,1)

img=cv2.imread('imagetest.jpg')
img=img[:,:,1]
img=im.fromarray(img)
img=resizeimage.resize('contain',img, [128, 160])
#img.resize([height, width],im.ANTIALIAS)
img=np.array(img)
img=img[:,:,0]

def homogenous (u):
    v=[u[0],u[1],1]
    return(v)
    
    
    
def vortex(K, depth_map, img):
    vortex=list()
    invK=np.linalg.inv(K)    
    for i in range(np.size(img,0)):
        for j in range(np.size(img,1)):
            u=[i,j]
            u=homogenous(u)
            g=np.dot(invK,u)*depth_map[i,j]
            vortex.append(np.concatenate((g,[img[i,j]])))
    return(vortex)        



vort1=vortex(newcameramtx, depth_map, img.T)
vort1=np.array(vort1)
vort1=np.absolute(vort1)            
vor1=vort1[0:20480, 0:3]
vor1[0:20480,0:2]=vor1[0:20480,0:2]/100
vor1[0:20480,2]=vor1[0:20480,2]/200
colors1=vort1[0:20480,3]
colors1=colors1.reshape([20480,1])
color=colors1
colors1=np.hstack([colors1, colors1])
colors1=np.hstack([color, colors1])

#########################################IMAGE 2####################################"
depth_map=cv2.imread('dispartitytest.png')
depth_map = cv2.cvtColor(depth_map,cv2.COLOR_BGR2GRAY)
depth_map=np.array(depth_map)
height=np.size(depth_map,0)
width=np.size(depth_map,1)

img=cv2.imread('imagetest.jpg')
img=img[:,:,1]
img=im.fromarray(img)
img=resizeimage.resize('contain',img, [128, 160])
#img.resize([height, width],im.ANTIALIAS)
img=np.array(img)
img=img[:,:,0]

vort2=vortex(newcameramtx, depth_map, img.T)
vort2=np.array(vort2)
vort2=np.absolute(vort2)            
vor2=vort2[0:20480, 0:3]
vor2[0:20480,0:2]=vor2[0:20480,0:2]/100
vor2[0:20480,2]=vor2[0:20480,2]/200
colors2=vort2[0:20480,3]
colors2=colors2.reshape([20480,1])
color=colors2
colors2=np.hstack([colors2, colors2])
colors2=np.hstack([color, colors2])

###################################IMAGE 3############################################
depth_map=cv2.imread('dispartitytest.png')
depth_map = cv2.cvtColor(depth_map,cv2.COLOR_BGR2GRAY)
depth_map=np.array(depth_map)
height=np.size(depth_map,0)
width=np.size(depth_map,1)

img=cv2.imread('imagetest.jpg')
img=img[:,:,1]
img=im.fromarray(img)
img=resizeimage.resize('contain',img, [128, 160])
#img.resize([height, width],im.ANTIALIAS)
img=np.array(img)
img=img[:,:,0]

vort3=vortex(newcameramtx, depth_map, img.T)
vort3=np.array(vort3)
vort3=np.absolute(vort3)            
vor3=vort3[0:20480, 0:3]
vor3[0:20480,0:2]=vor3[0:20480,0:2]/100
vor3[0:20480,2]=vor3[0:20480,2]/200
colors3=vort3[0:20480,3]
colors3=colors3.reshape([20480,1])
color=colors3
colors3=np.hstack([colors3, colors3])
colors3=np.hstack([color, colors3])





colors=np.vstack([colors1, colors2, colors3])
vor=np.vstack([vor1,vor2, vor3])

##########################################test .PLY

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar blu
property uchar green
end_header
'''

def write_ply(fn, vor, colors):
    verts = np.hstack([vor, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

write_ply('out1.ply', vor, colors)
print('%s saved' % 'out1.ply')
