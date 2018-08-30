import numpy as np
import progressbar
import cv2
from Drawer import Drawer
from PIL import Image

DATA_DIR = '/home/viki/PFE/orbslam_frames/RGBD_example/'
TMU_DIR = DATA_DIR + 'rgbd_dataset_freiburg1_xyz/'
IDV_DIR = DATA_DIR + 'orbslam/'


class Reconstruction:
    __imageHeight = 480
    __imageWidth = 640
    __mtx = np.array([
        [620.423,0,327.327],
        [0,620.95,239.649],
        [0,0,1]
        ])
    __Ms_keyframe = {}
    __Ms = []
    scalingFactor = 5000
    __max_depth = 0.6
    
    def __init__(self, frame_doc, out):
        self.__frame = frame_doc
        self.__out_doc = out
        self.drawer = Drawer()
    
    def set_max_depth(self, num):
        self.__max_depth = num
        
    def load_camera_pose(self, traj_doc):
        keyframe_file = open(traj_doc+'/KeyFrameTrajectoryMatrix.txt', 'r') 
        keyframe_matrix_lines = keyframe_file.readlines() 
        for matrix_line in keyframe_matrix_lines:
            matrix_list = matrix_line.split()
            matrix = np.zeros((3,4))
            rr = np.zeros((3,3))
            matrix[:,3] = [matrix_list[1], matrix_list[2], matrix_list[3]]
            rr[0,0:3] = [matrix_list[4],matrix_list[5],matrix_list[6]]
            rr[1,0:3] = [matrix_list[7],matrix_list[8],matrix_list[9]]
            rr[2,0:3] = [matrix_list[10],matrix_list[11],matrix_list[12]]
            matrix[:,3] = matrix[:,3]/0.001
            matrix[0:3,0:3] = rr.T
            self.__Ms_keyframe[matrix_list[0]] = matrix
        print(" Camera pose matirx loaded.")
        
    def load_camera_pose_tmu(self, traj_doc):
        keyframe_file = open(traj_doc+'/KeyFrameTrajectory.txt', 'r') 
        keyframe_matrix_lines = keyframe_file.readlines() 
        for matrix_line in keyframe_matrix_lines:
            matrix_list = matrix_line.split()
            matrix = transform34(matrix_list)
            self.__Ms_keyframe[matrix_list[0]] = matrix
        print(" Camera pose matirx loaded.")
    """
    def vert_new(self, img,D, T, if_inv = False, rate = 1):
        l = img.shape[0]
        h = img.shape[1]
        t = 0
        temp = np.zeros((l*h, 6))
        T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
        if if_inv:
            T = np.linalg.inv(T)

        for i in range(0,l,rate):
            for j in range(0,h,rate):
                dep = D[i,j]
                if 0.1 < dep < self.__max_depth:
                    #camera caliberation
                    img_p = np.array([i, j, 1])
                    real_p = np.dot(np.linalg.inv(self.__mtx),(img_p)*dep)
                    real_p = np.append(real_p, 1)
                    #real_p[1] = - real_p[1]
                    aa = np.dot(T, real_p).reshape(1,4)
                    temp[t,0:3]  = aa[0,0:3]/aa[0,3]
                    #temp[t,1] = -temp[t,1]
                    temp[t,3:6] = img[i,j]
                    t = t+1
        return temp[0:t,:]
    """
    def vert_new(self, rgb_file, depth_file, transform, downsample = 1, pcd = False):
        rgb = Image.open(rgb_file)
        depth = Image.open(depth_file)
        points = []    
        for v in range(0,rgb.size[1],downsample):
            for u in range(0,rgb.size[0],downsample):
                color = rgb.getpixel((u,v))
                Z = depth.getpixel((u,v)) / self.scalingFactor
                if Z==0 or Z > self.__max_depth: continue
                X = (u - self.__mtx[0,2]) * Z / self.__mtx[0,0]
                Y = (v - self.__mtx[1,2]) * Z / self.__mtx[1,1]
                vec_org = np.matrix([[X],[Y],[Z],[1]])
                if pcd:
                    points.append(struct.pack("fffI",vec_org[0,0],vec_org[1,0],
                                vec_org[2,0],color[0]*2**16+color[1]*2**8+color[2]*2**0))
                else:
                    vec_transf = np.dot(transform,vec_org)
                    points.append("%f %f %f %d %d %d 0\n"%(vec_transf[0,0],
                                vec_transf[1,0],vec_transf[2,0],color[0],color[1],color[2]))
        return points

    def write_ply_list(self, ply_file,points):
        file = open(ply_file,"w")
        file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            '''%(len(points),"".join(points)))
        file.close()
        #print "Saved %d points to '%s'"%(len(points),ply_file)

    
    def write_ply_array(self,fn,verts ):
        ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d')
      
            
    def make_ply(self, rate=1):
        length_m=len(self.__Ms_keyframe)
        bar = progressbar.ProgressBar(maxval=length_m, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        i = 0
        idx_old = 0
        for idx, matrix in self.__Ms_keyframe.items():
            idx = int(float(idx))
            if idx > idx_old + 10 or idx_old == 0:
                idx1 = idx
                #image = cv2.imread(self.__frame + '/rgb/' + str(idx).zfill(6) + '.png')
                #depth = cv2.imread(self.__frame + '/depth/' + str(idx).zfill(6) + '.png')[:,:,0]
                #depth = np.load(self.__frame + '/depth_npy/' + str(idx).zfill(6) + '.npy')
                #if idx > 350:
                #    idx1 = idx + 651
                image = self.__frame + '/rgb/' + str(idx1).zfill(6) + '.png'
                depth = self.__frame + '/depth/' + str(idx1).zfill(6) + '.png'
                points = self.vert_new(image, depth, matrix, rate)
                self.write_ply_list(self.__out_doc + '/' +str(idx)+'.ply', points)
                idx_old = int(idx)
            #print('  finished one image')
            bar.update(i+1)
            i = i + 1
        bar.finish()
        #print('----- finished -----')
        
    def draw_camera(self):
        res0 = np.zeros((1,6))
        poss0 = np.zeros(3)
        for idx, matrix in self.__Ms_keyframe.items():
            res, poss1 = self.drawer.draw_camera(matrix)
            res0 = np.concatenate((res0,res), axis=0)
        """"
        for idx in self.main_loop_idx:
            Mss = self.Ms_multi[idx]
            for M in Mss:
                res, poss1 = self.drawer.draw_camera(M)
                lene = self.drawer.draw_line(poss0, poss1)
                poss0 = poss1
                res0 = np.concatenate((res0,res,lene), axis=0)
        """
        self.write_ply_array(self.__out_doc + '/camera.ply', res0)
        print(" --- camera drawn")
        return 0


def transform34(l):
    t = l[1:4]
    factor = 1
    t = np.array([float(t[0]),float(t[1]),float(t[2])])*factor
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    EPS = np.finfo(float).eps * 4.0
    if nq < EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2])
        ), dtype=np.float64)
