from utils_minecraft import *

res = np.loadtxt('fin_1.ply')
_, list_direction = from_ply_to_minecraft(res, 0.1)
xyz, blocks = minecraft_xyz_block( list_direction , 3)

np.save('xyz.npy',xyz)
np.save('blocks.npy', blocks)
