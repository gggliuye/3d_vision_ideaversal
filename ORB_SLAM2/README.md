try to make 3d reconstruction based on ORB SLAM2 
[ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)


### orbslam_rs
allow to use orbslam2 with IntelRealsense in real time

### Delaunay
[Delaunay](https://github.com/gggliuye/3d_vision/blob/master/ORB_SLAM2/Delaunay.ipynb)
- getting the ORB feature points from ORB_SLAM2 and try to build model delaunay based on these points.

### Reconstruction
[Reconstruction](https://github.com/gggliuye/3d_vision/blob/master/ORB_SLAM2/make_ply.ipynb)
- usine the keyframe trajectory outputs
- build reconstruction from these keyframes
- todo make dense reconstruction or fusion


#### about install orbslam2
- error: usleep is not declared in this scope
find these files, and add the following codes to it:

#include <unistd.h>

#include <stdio.h>

#include <stdlib.h>

- for ROS add th
change into the following lines in cmakelist of ROS:

set(LIBS

${OpenCV_LIBS}

${EIGEN3_LIBS}

${Pangolin_LIBRARIES}

${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so

${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so

${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so

-lboost_system # add this line

)
