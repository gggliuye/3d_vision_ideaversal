import sys
import cv2
import numpy as np
import glob
import os
from math import sqrt
from numpy import *
from shapely.geometry import Polygon

class Drawer:

    def __init__(self):
        self.result = None

    def draw_camera(self, T,inv = True):
        T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
        if inv:
            T = np.linalg.inv(T)
        T = T[0:3,:]
        center = T[0:3,3]

        pos = []
        s = 12
        pos.append([0,0,1.5*s,1])
        pos.append([s,s,0,1])
        pos.append([s,-s,0,1])
        pos.append([-s,-s,0,1])
        pos.append([-s,s,0,1])

        pos = np.array(pos)
        n_cam = (np.dot(T,pos.T)).T

        res = np.zeros((1,6))
        for i in range(1,5):
            res1 = self.draw_line(n_cam[0],n_cam[i], 'camera')
            j = 1 if i==4 else i+1
            res2 = self.draw_line(n_cam[i],n_cam[j], 'camera')
            res = np.concatenate((res,res1,res2), axis=0)
        return res[1:], center

    def draw_polygen(self, top,z_max,z_min, off_set,  color):
        res = np.zeros((1,6))
        for i in range(len(top)):
            j = 0 if i+1==len(top) else i+1
            res1 = self.draw_line(np.array([z_min,top[i][0]+off_set[0],top[i][1]+off_set[1]]),np.array([z_min,top[j][0]+off_set[0],top[j][1]+off_set[1]]),color)
            res2 = self.draw_line(np.array([z_max,top[i][0],top[i][1]]),np.array([z_max,top[j][0],top[j][1]]),color)
            res3 = self.draw_line(np.array([z_min,top[i][0]+off_set[0],top[i][1]+off_set[1]]),np.array([z_max,top[i][0],top[i][1]]),color)
            res = np.concatenate((res,res1,res2,res3), axis=0)
        return res[1:]

    def draw_bbx(self, top, z_max, z_min, color=0, off_set = np.zeros(2)):
        # set color
        if color == 0:
            temp = np.random.randint(255)
            color = str(temp)+'_'+str(temp)+'_'+str(temp)

        res = self.draw_polygen(top,z_max,z_min,off_set,color)
        return res



    def draw_line(self, x,y, color='255_255_0'):
        x = x.reshape(1,3)
        y = y.reshape(1,3)
        n = sqrt( (x[0,0] - y[0,0])**2 + (x[0,1] - y[0,1])**2+(x[0,2] - y[0,2])**2 )/0.5
        n = max(int(n),2)
        res = np.zeros((n ,6),dtype = np.float32)
        des = (y - x)/(n-1)
        for i in range(n):
            res[i, 0:3] = x + des*i
            if color == 'camera':
                res[i, 3:6] = np.array([0,255,255])
            else:
                tt = color.split("_")
                res[i, 3] = int(tt[0])
                res[i, 4] = int(tt[1])
                res[i, 5] = int(tt[2])
        return res
