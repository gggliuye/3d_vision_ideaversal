import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import signal


def get_camera_matrix(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #print(rvecs, tvecs)
    return mtx, dist,imgpoints

def trans_matrix(r,t):
    # T (ki to t), rotate and transport matrix
    (a,b,c) = (r[0][0], r[1][0], r[2][0])
    (x,y,z) = (t[0][0],t[1][0],t[2][0])
    Rz = np.matrix([[1,0,0],
          [0, np.cos(a), -np.sin(a)],
         [0, np.sin(a), np.cos(a)]])
    Ry = np.matrix([[np.cos(b), 0 , np.sin(b)],
          [0 , 1, 0],
          [-np.sin(b), 0 , np.cos(b)]])
    Rx = np.matrix([[np.cos(c), -np.sin(c), 0],
         [np.sin(c), np.cos(c), 0],
         [0,0,1]])
    R = np.dot(Rz,np.dot(Ry,Rx))
    T = np.concatenate((R, np.matrix([[x],[y],[z]])), axis=1)

    return T

def transpose_matrix(fname, mtx, dist):
    objp = np.zeros((5*5,3), np.float32)
    objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (5,5),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    else:
        return False
    T = trans_matrix(rvecs, tvecs)

    return T

def vert(img,D, T,K, inv = True):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)
    K = np.eye(3)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 10 < D[i,j]< 200:
                u = np.array([i,j, D[i,j], 1])
                aa = np.dot(T, u)
                x = np.zeros(3)
                x[0:3] = aa[0:3]/aa[3]
                x = np.dot(K, x)

                temp[t,0] = x[0]
                temp[t,1] = x[1]
                temp[t,2] = x[2]
                temp[t,3:6] = img[i,j]
                t = t+1
    return temp

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar blue
property uchar green
end_header
'''
def write_ply(fn,verts ):
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')
