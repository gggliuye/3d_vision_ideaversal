#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<opencv2/core/core.hpp>

#include <geometry_msgs/Pose.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>

#include"../../../include/System.h"
#include"../../../include/Converter.h"

using namespace std;

class ImageGrabber
{
private:
    ORB_SLAM2::System* mpSLAM;
    ros::Publisher* camera_pose_pub;
    tf::TransformBroadcaster br;

public:
    ImageGrabber(ORB_SLAM2::System* pSLAM, ros::Publisher* camera_poses):mpSLAM(pSLAM), camera_pose_pub(camera_poses){}
    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ros::NodeHandle nh;

    //ros::Publisher pub_com = nh.advertise<geometry_msgs::Pose>("camera_pose", 1);
    ros::Publisher pub_com = nh.advertise<geometry_msgs::TransformStamped>("camera_pose_tf", 1);

    ImageGrabber igb(&SLAM, &pub_com);

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat Tcw = mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());

    if (Tcw.empty()) {
        cout << " ---- Lost track ----" << endl;
        return;
    }

    //tf broad cast the translation matrix
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

    vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);

    geometry_msgs::TransformStamped camera_pose;
    camera_pose.header = cv_ptrD->header;
    camera_pose.header.frame_id = "world";
    camera_pose.child_frame_id = "frame";

    camera_pose.transform.translation.x = twc.at<float>(0, 0);
    camera_pose.transform.translation.y = twc.at<float>(0, 1);
    camera_pose.transform.translation.z = twc.at<float>(0, 2);
    camera_pose.transform.rotation.x = q[0];
    camera_pose.transform.rotation.y = q[1];
    camera_pose.transform.rotation.z = q[2];
    camera_pose.transform.rotation.w = q[3];

    camera_pose_pub->publish(camera_pose);

    /*
    geometry_msgs::Pose camera_pose;
    camera_pose.position.x = twc.at<float>(0, 0);
    camera_pose.position.y = twc.at<float>(0, 1);
    camera_pose.position.z = twc.at<float>(0, 2);
    camera_pose.orientation.x = q[0];
    camera_pose.orientation.y = q[1];
    camera_pose.orientation.z = q[2];
    camera_pose.orientation.w = q[3];

    camera_pose_pub->publish(camera_pose);
    */


    tf::Transform transform;
    transform.setOrigin(tf::Vector3(twc.at<float>(0, 0), twc.at<float>(0, 1), twc.at<float>(0, 2)));
    tf::Quaternion quaternion(q[0], q[1], q[2], q[3]);
    transform.setRotation(quaternion);

    br.sendTransform(tf::StampedTransform(transform, ros::Time(cv_ptrD->header.stamp.toSec()), "world", "frame"));

}


