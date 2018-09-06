#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<librealsense2/rs.hpp>
#include<opencv2/core/core.hpp>
#include<System.h>
#include <unistd.h>

using namespace std;

float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);



int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    int nImages = stoi(argv[4]);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images to be in the sequence: " << nImages << endl << endl;

    // for start realsense pipeline
    cout << endl << "-------" << endl;
    cout << "Start realsense pipeline ..." <<endl;
    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
    rs2::colorizer color_map;

    rs2::pipeline pipe;

    //Create a config and configure the pipeline to stream
    //different resolutions of color and depth streams
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_YUYV, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    //cfg.enable_stream(RS2_STREAM_INFRARED,1, 640, 480, RS2_FORMAT_Y8, 30);
    //Start streaming
    rs2::pipeline_profile profile = pipe.start(cfg);

    rs2_stream align_to = find_stream_to_align(profile.get_streams());
    rs2::align align(align_to);

    //Getting the depth sensor's depth scale (see rs-align example for explanation)
    float depth_scale = get_depth_scale(profile.get_device());
    cout << "  -  depth scale is : " << depth_scale << endl;
    cout << "Intel realsense ready. " <<endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
	// get image and depthmap from realsense
	rs2::frameset data = pipe.wait_for_frames();
	auto processed = align.process(data);
        rs2::video_frame other_frame = processed.first_or_default(align_to);

        rs2::frame aligned_depth_frame = processed.get_depth_frame();
	rs2::frame aligned_color_frame = processed.get_color_frame();
	//aligned_depth_frame = color_map.process(aligned_depth_frame);

    	cv::Mat matDepth = cv::Mat(cv::Size(640,480), CV_16UC1, (void*)aligned_depth_frame.get_data(), cv::Mat::AUTO_STEP);
	cv::Mat matColor = cv::Mat(cv::Size(640,480), CV_8UC2, (void*)aligned_color_frame.get_data(), cv::Mat::AUTO_STEP);
	vector<cv::Mat> channels(2);
	cv::split(matColor, channels);
	matColor = channels[0];
	
	cv::imwrite(string(argv[3])+"/depth/"+to_string(ni)+".png", matColor);
	cv::imwrite(string(argv[3])+"/rgb/"+to_string(ni)+".png", matDepth);

        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/depth/"+to_string(ni)+".png",CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/rgb/"+to_string(ni)+".png",CV_LOAD_IMAGE_UNCHANGED);
        double tframe = ni;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T = 0.1;

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");  

    // Stop all threads
    SLAM.Shutdown(); 

    return 0;
}


float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}



