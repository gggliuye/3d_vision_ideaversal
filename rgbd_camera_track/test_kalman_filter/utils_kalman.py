import numpy as np
from pykalman import UnscentedKalmanFilter, AdditiveUnscentedKalmanFilter
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

# time between two images
dt = 1/20
# video image size
video_height = 480
video_width = 640
# camera parameteres
lens = 3.6
sensor_size = [3.67,2.74]


def points_from_dist_angle(dist, angle):
    num = len(dist)
    points = np.zeros((num+1,2))
    ang = 0
    for i in range(num):
        ang = ang + angle[i]
        points[i+1,0] = points[i,0] - dist[i]*np.sin(ang*np.pi/180)
        points[i+1,1] = points[i,1] + dist[i]*np.cos(ang*np.pi/180)
    return points

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def rotate_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # list of distances between frames
    distances = []
    while(1):
        for j in range(3):
            ret,frame = cap.read()
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            print('   -  no feature point detected, process break.')
            break
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # calcule the move distance
        distance = []
        for i in range(len(good_new)):
            distance.append(good_new[i][0]-good_old[i][0])
        distance = reject_outliers(np.asarray(distance))
        distance = np.mean(distance)
        distances.append(distance)
        # update old frame
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    cap.release()
    #print(distances)
    rotates = np.asarray(distances)[0:-1]
    return np.sum(rotates)*54/video_width

def rotate_and_translation_from_video(video_file):
    cap = cv2.VideoCapture(video_file)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # list of distances between frames
    rotates = []
    ratio_translate = []
    while(1):
        for j in range(3):
            ret,frame = cap.read()
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            print('   -  no feature point detected, process break.')
            break
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # calcule the move distance
        rotate = []
        ratio = []
        for i in range(len(good_new)):
            theta = np.arctan2(abs(good_old[i][1]-video_height/2), (good_old[i][0]-video_width/2))
            dx = (good_new[i][1]-good_old[i][1])/np.sin(theta)
            radio = abs(dx)/(np.sqrt((good_old[i][1]-video_height/2)**2
                           + (good_old[i][0]-video_width/2)**2))
            rotate.append((good_new[i][0]-good_old[i][0]) -
                          abs(good_new[i][1]-good_old[i][1])*np.tan(theta))
            ratio.append(radio)
        ratio = reject_outliers(np.asarray(ratio))
        ratio = np.mean(ratio)
        ratio_translate.append(ratio)

        rotate = reject_outliers(np.asarray(rotate))
        rotate = np.mean(rotate)
        rotates.append(rotate)
        # update old frame
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    cap.release()
    rotates = np.asarray(rotates)[1:-1]
    ratio_translate = np.nan_to_num(np.asarray(ratio_translate)[1:-1])
    return np.sum(ratio_translate)*30, np.sum(rotates)*54/video_width

def transition_function(state, noise):
    state[0] = state[0] + dt*state[3]
    state[1] = state[1] + dt*state[4]
    state[2] = state[2] + dt*state[5]
    return state + noise

def observation_function(state, noise):
    n = int((len(state) - 6)/2)
    #observation = np.zeros(2*n)
    observation = np.zeros(n)
    for i in range(n):
        dx = state[2*i+6] - state[0]
        dy = state[2*i+7] - state[1]
        theta = np.arctan2(dx,dy) - state[2]
        #if -np.pi/4 < theta < np.pi/4:
        #    observation[2*i+1] =  theta
        #    observation[2*i] = np.sqrt(dx*dx + dy*dy)
        #observation[2*i+1] =  theta
        #observation[2*i] = np.sqrt(dx*dx + dy*dy)
        observation[i] =  theta
    return observation + noise


def additive_transition_function(state):
    xx = len(state)
    return transition_function(state, np.zeros(xx))

def additive_observation_function(state):
    xx = int((len(state) - 6)/2)
    return observation_function(state, np.zeros(xx))


def get_feature_points_from_video(videofile):
    cap = cv2.VideoCapture(videofile)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    init_image = old_frame
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    # initial list of the points
    points = []
    points.append(p0)
    # calcule total number of images, 10 images per second
    #total_frames = cap.get(7)
    #number = int(np.floor(total_frames/3))
    #print('we will take ',number, ' images from the video.')
    #for i in range(number):
    while True:
        #cap.set(1, 3*i)
        #ret,frame = cap.read()
        for j in range(3):
            ret,frame = cap.read()
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # update points in the list
        for i in range(len(points)):
            points[i] = points[i][st==1].reshape(-1,1,2)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        # append p0 to the list
        points.append(p0)
    cap.release()
    print('feature points shape from optical flow is ', np.shape(points))
    return np.asarray(points), init_image

def delete_pts_centre(points):
    num_fea = points.shape[1]
    idx = []
    thre = 120
    for i in range(num_fea):
        distance = np.sqrt((points[0,i,0,1]-video_height/2)**2
                           + (points[0,i,0,0]-video_width/2)**2)
        print(distance)
        if distance > thre:
            idx.append(i)
    # delete
    print(idx)
    n_points = np.zeros((points.shape[0],(len(idx)),1,2))
    t = 0
    for i in idx:
        n_points[:,t,0,:] = points[:,i,0,:]
        t = t+1
    return n_points

def feature_points_to_observation_and_initilization(points, v0, dep0):
    # set camera calibration matrix
    #points = delete_pts_centre(points)
    print('new points is ',points.shape)
    fx = video_width*lens/sensor_size[0]
    fy = video_height*lens/sensor_size[1]
    K = np.array([[fy, 0, video_height/2],
                 [0, fx, video_width/2],
                 [0,0,1]])
    print(K)
    # set state of robot
    robot_param = np.zeros(6)
    robot_param[4] = v0
    # get the numbre of features and observations
    num_feature = points.shape[1]
    num_obser = points.shape[0] - 1
    # set the first observation feature points as initilizaion
    init_feature = np.zeros(2*num_feature)
    init = points[0].reshape(-1,2)
    for i in range(num_feature):
        dep = dep0[int(init[i,1]), int(init[i,0])]
        img_p = np.array([init[i,1], init[i,0], 1])
        real_p = np.dot(np.linalg.inv(K),(img_p)*dep)
        init_feature[2*i] = real_p[1]
        init_feature[2*i+1] = real_p[2]
    initilization = np.concatenate([robot_param, init_feature])
    # set obseravations of feature points
    observation = []
    for i in range(num_obser):
        obs_feature = points[i+1].reshape(-1,2)
        obs = np.zeros(num_feature)
        for j in range(num_feature):
            dx = (obs_feature[j,0]-video_width/2)*sensor_size[0]/video_width
            obs[j] = np.arctan2(dx,lens)
        observation.append(obs)

    return initilization, observation

def kalman_filter(video_file, dep0, v0):
    # get feature points from video
    points, init_image = get_feature_points_from_video(video_file)
    i, o = feature_points_to_observation_and_initilization(points, v0, dep0)
    # number of the features points
    observations = np.asarray(o)
    n = observations.shape[1]
    print('initilization shape is ', i.shape)
    print('observation shape is ', observations.shape)
    # initialize the vovariance and mean
    transition_covariance = np.eye(2*n+6)
    random_state = np.random.RandomState(0)
    observation_covariance = np.eye(n) + abs(random_state.randn(n, n) * 0.005)

    initial_state_mean = i
    covariance_init = random_state.randn(2*n+6, 2*n+6) * 0.2
    covariance_init[0:3,0:3] = 0.005
    initial_state_covariance =  np.eye(2*n+6) + abs(covariance_init)
    # set Unscented kalman filter
    kf = UnscentedKalmanFilter(
        transition_function, observation_function,
        transition_covariance, observation_covariance,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
        )
    """
    kf = AdditiveUnscentedKalmanFilter(
        additive_transition_function, additive_observation_function,
        transition_covariance, observation_covariance,
        initial_state_mean, initial_state_covariance
        )
    """
    # get result
    filtered_state_estimates = kf.filter(observations)
    smoothed_state_estimates = kf.smooth(observations)
    return filtered_state_estimates, smoothed_state_estimates

def from_KF_result_to_points(estimate):
    num = int((len(estimate) - 6)/2)
    x = np.zeros(num)
    y = np.zeros(num)
    for i in range(num):
        x[i] = estimate[6+2*i]
        y[i] = estimate[6+2*i+1]
    return x,y


def calu_dep(imagefile,depfile):
    ckptfile = 'F:/pfe/cnn_slam/FCRN-DepthPrediction-master/ckpt/NYU_FCRN.ckpt'
    my_file = Path(depfile)
    if not my_file.is_file():
        os.system("python predict.py "+ ckptfile + " " +imagefile+" "+depfile )
    return 0


def calucate_dep(init_image):
    cv2.imwrite('temp.jpg',init_image)
    a = calu_dep('temp.jpg','dep.npy')
    return np.load('dep.npy')
