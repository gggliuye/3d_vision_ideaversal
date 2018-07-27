import mcpi.minecraft as minecraft
import mcpi.block as block
import mcpi.entity as entity
import math
import server
import sys
import numpy as np


mc = minecraft.Minecraft()
mc.postToChat("Hello world!")
x, y, z = mc.player.getPos()

xyz = np.load('ideaversal/xyz.npy')
blocks = np.load('ideaversal/blocks.npy')

for i in range(80):
    for j in range(80):
        mc.setBlock(x+i-25,y-4, z+j-25,  block.DIRT)

siww = 8
for i in range(len(blocks)):
    xx = x + xyz[i,0]
    yy = y + xyz[i,1]
    zz = z + xyz[i,2]
    #bl = blocks[i]
    #mc.setBlock(xx,yy, zz, int(bl[0]), int(bl[1]))
    if (-siww<xyz[i,0]<siww and -siww<xyz[i,2]<siww):
        mc.setBlock(xx,yy, zz, block.REDSTONE_BLOCK )
    else:
        mc.setBlock(xx,yy, zz, block.QUARTZ_BLOCK)



mc.postToChat("Finish build !!")
