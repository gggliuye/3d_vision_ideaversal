/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

#include<sl_zed/Camera.hpp>

using namespace std;
using namespace sl;

bool save_map = false;

cv::Mat slMat2cvMat(Mat& input);

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings" << endl;
        return 1;
    }
   
    /////////////////////////////
    /// connecting ZED camera ///
    cout << " -- Connecting ZED camera. -- " << endl;

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
    init_params.coordinate_units = UNIT_CENTIMETER; // unit as centimeter so the depthh scale should be 100
    //if (argc > 1) init_params.svo_input_filename.set(argv[1]);

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;
    Resolution image_size = zed.getResolution();

    int new_width = image_size.width ;
    int new_height = image_size.height ;

    cout << "ZED MINI camera connected , resolution is :  " << new_width << " * " << new_height << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Create SLAM system. It initializes all system threads and gets ready to process frames.//
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true, false);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    // Main loop
    Mat image_zed(new_width, new_height, MAT_TYPE_8U_C4);  // view images have the type : 8 unsigned int with 4 channel
    cv::Mat image_ocv = slMat2cvMat(image_zed);
    Mat depth_image_zed(new_width, new_height, MAT_TYPE_32F_C1);  // the measure depth type is 32 float and one channel
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);

    int time_i = 0;
    //char key = ' ';
    while(zed.grab(runtime_parameters) == SUCCESS)
    {
	zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU, new_width, new_height);
        zed.retrieveMeasure(depth_image_zed, MEASURE_DEPTH, MEM_CPU, new_width, new_height);
        // zed.retrieveImage(depth_image_zed, VIEW_DEPTH, MEM_CPU, new_width, new_height);

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(image_ocv,depth_image_ocv, time_i);
        time_i = time_i + 1;

    }

    // Stop all threads
    SLAM.Shutdown();

    if(save_map) 
    {
       SLAM.saveMapToFile();
    }
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}


/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

