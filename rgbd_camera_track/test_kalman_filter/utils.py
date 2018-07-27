import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import os
from pathlib import Path
from utils_kalman import *

# global variables
newHeight = 128
newWidth = 160
sensor_size = [3.67,2.74]
lens = 3.6
ckptfile = 'F:/pfe/cnn_slam/FCRN-DepthPrediction-master/ckpt/NYU_FCRN.ckpt'
fx = newHeight*lens/sensor_size[0]
fy = newWidth*lens/sensor_size[1]
mtx = np.array([[fy, 0, newHeight/2],
             [0, fx, newWidth/2],
             [0,0,1]])

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
def write_ply(fn,verts ):
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def trans_matrix(r,t):
    # T (ki to t), rotate and transport matrix
    r = r.ravel()
    t = t.ravel()
    (a,b,c) = (r[0], r[1], r[2])
    (x,y,z) = (t[0], t[1], t[2])
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


def vert_25_06(img,D, T,K, inv = True):
    l = img.shape[0]
    h = img.shape[1]
    t = 0
    temp = np.zeros((l*h, 6))
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    if inv:
        T = np.linalg.inv(T)
    for i in range(l):
        for j in range(h):
            dep = D[i,j]
            if 0.1<dep< 200000:
                img_p = np.array([i, j, 1])
                real_p = np.dot(np.linalg.inv(K),(img_p)*dep)
                real_p = np.append(real_p, 1)
                aa = np.dot(T, real_p).reshape(1,4)
                temp[t,0:3] = aa[0,0:3]/aa[0,3]
                temp[t,3:6] = img[i,j]
                t = t+1
    return temp[0:t,:]

def vert_alphabot_fcn(depth,mtx):
    num = 20
    dep = np.zeros(20)
    kt = np.linalg.inv(mtx)
    for i in range(num):
        temp = depth[127-i,:]
        temp = reject_outliers(temp)
        dep[i] = np.mean(temp)
    #dep[0] = depth.min() + 0.4
    constant = (kt[0,0]*127 + kt[0,2])*dep[0]

    t = np.zeros(19)
    for i in range(1,num):
        if (dep[i]-dep[0]) == 0:
            t[i-1] = 20
        else:
            t[i-1] = (constant/(kt[0,0]*(127-i) + kt[0,2]) - dep[0])/(dep[i]-dep[0])
    for i in range(4):
        t = reject_outliers(t)
    #variance = np.sum(np.square(t- np.mean(t)))
    #print('    --- variance is '+ str(variance))
    return np.mean(t), dep[0]

def remake_depth_projecting_pts_to_floor(D,K):
    l = D.shape[0]
    h = D.shape[1]
    ktt,d0 = vert_alphabot_fcn(D, K)
    new_dep = D
    for i in range(l):
        for j in range(h):
            dep = ktt*(D[i,j] - d0) + d0
            new_dep[i,j] = dep
    return new_dep

def get_matrix_from_video(video_file):
    ang1 = rotate_from_video(video_file)
    dis, ang2 = rotate_and_translation_from_video(video_file)
    print(ang1, dis,ang2)
    """
    if ang2 < 10:
        dis = 0
        ang = ang1
    else:
        ang = ang2
        dis = (dis*0.2 + 45*0.8)
    """
    rotate = ['keys/videos/video804.h264','keys/videos/video807.h264']
    print(video_file)
    if video_file in rotate:
        print('rotation')
        dis = 0
        ang = ang1
        if np.isnan(ang):
            ang = 70

    else:
        ang = ang2
        dis = (dis*0.2 + 45*0.8)
    dis = dis/100
    r = np.array([ang*np.pi/180,0,0])
    t = np.array([0,dis*np.sin(ang*np.pi/180),dis*np.cos(ang*np.pi/180)])
    return trans_matrix(r,t)

def mut_trans(M1, M2):
    M1 = np.concatenate((M1, np.array([[0,0,0,1]])), axis=0)
    M2 = np.concatenate((M2, np.array([[0,0,0,1]])), axis=0)
    M = np.dot(M1,M2)
    return M[0:3,:]

def add_image_25_06(imagefile, depfile, videofile, old_res, iteration=0,seg = False):
    #print('Start matching the image ', imagefile)

    image0 = cv2.imread(imagefile)
    dep = np.load(depfile).reshape((newHeight,newWidth))
    #dep = remake_depth_projecting_pts_to_floor(dep,mtx)
    l = iteration

    Ms = glob.glob('keys/Ms/*.npy')
    M_old = np.eye(3,4)
    for mm_f in Ms:
        mm = np.load(mm_f)
        M_old = mut_trans(M_old, mm)
    M = M_old
    print(M)
    M_new = get_matrix_from_video(videofile)
    np.save('keys/Ms/'+str(l+1) +'.npy' , M_new)
    image = cv2.resize(image0, (newWidth, newHeight))

    # "image" is the new added image "dep" is its depth map
    if old_res is None:
        res_final = vert_25_06(image, dep, M,mtx, False)
    else:
        res_new_image = vert_25_06(image, dep, M,mtx, False)
        res_final = np.concatenate((old_res,res_new_image), axis=0)

    dirr = 'keys/'+str(iteration)
    if not os.path.exists(dirr):
        os.makedirs(dirr)
    cv2.imwrite(dirr+'/'+str(l+1) +'.jpg',image0)
    np.save(dirr+'/'+str(l+1) +'.npy',dep)

    return res_final
