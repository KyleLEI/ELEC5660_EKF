#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>

#include "EKF.h"

EKF* ekf;
using namespace std;
using namespace Eigen;
ros::Publisher odom_pub;
MatrixXd Q = MatrixXd::Identity(15, 15);
MatrixXd Rt = MatrixXd::Identity(9,9);
double imu_time, last_imu_time=-1.0; 

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    //your code for propagation
    if(last_imu_time<0){//not initialized
        last_imu_time = msg->header.stamp.toSec();
        return;
    }
    imu_time = msg->header.stamp.toSec();
    VectorXd u(6);
    u<< msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z,
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z;
    //cout<<"Predict =\n"<<ekf->predict(u,imu_time-last_imu_time)<<endl;
    last_imu_time = imu_time;
}

//Rotation from the camera frame to the IMU frame
Vector3d T_ic;
Eigen::Matrix3d R_ic;
void odom_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    //your code for update
    //For part 1
    // camera position in the IMU frame = (0.05, 0.05, 0)
    // camera orientaion in the IMU frame = Quaternion(0, 1, 0, 0); w x y z, respectively
    //					   RotationMatrix << 1, 0, 0,
    //							             0, -1, 0,
    //                                       0, 0, -1;
    
    // For part 2 & 3
    // camera position in the IMU frame = (0.1, 0, 0.03)
    // camera orientaion in the IMU frame = Quaternion(0, 0, 0, 1); w x y z, respectively
    //                     RotationMatrix << -1, 0, 0,
    //                                       0, -1, 0,
    // 
    VectorXd z(9);

    Vector3d T_cw;
    Matrix3d R_cw;
    T_cw<<  msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z;
    R_cw = Quaterniond(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z
    ).toRotationMatrix();
    std::cout<<"R_cw = \n"<<R_cw<<std::endl;
    std::cout<<"T_cw = \n"<<T_cw<<std::endl;

    Vector3d T_wc = -1*T_cw;
    Matrix3d R_wc = R_cw.transpose();
    Vector3d T_ci = -1*T_ic;
    Matrix3d R_ci = R_ic.transpose();
   
    Matrix3d R_wi = R_wc*R_ci;
    std::cout<<"R_wi = \n"<<R_wi<<std::endl;
    Vector3d T_wi = R_wc*T_ci+T_wc;
    std::cout<<"T_wi = \n"<<T_wi<<std::endl;

    double roll = asin(R_wi(2,1));
    double yaw = atan2(-R_wi(0,1)/cos(roll),R_wi(1,1)/cos(roll));
    double pitch = atan2(-R_wi(2,0)/cos(roll),R_wi(2,2)/cos(roll));
    
    z<< T_wi.x(),
        T_wi.y(),
        T_wi.z(),
        roll,
        pitch,
        yaw,
        0,
        0,
        0;
    std::cout<<"z = \n"<<z<<std::endl;
    ekf->update(z);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ekf");
    ros::NodeHandle n("~");
    ros::Subscriber s1 = n.subscribe("imu", 1000, imu_callback);
    ros::Subscriber s2 = n.subscribe("tag_odom", 1000, odom_callback);
    odom_pub = n.advertise<nav_msgs::Odometry>("ekf_odom", 100);
    R_ic = Quaterniond(0, 1, 0, 0).toRotationMatrix();
    T_ic<<0.05,0.05,0;
    cout << "R_ic =\n" << R_ic << endl;
    cout<<"T_ic = \n"<<T_ic<<endl;
    // Q imu covariance matrix; Rt visual odomtry covariance matrix
    // You should also tune these parameters
    /*
    Q.topLeftCorner(6, 6) = 0.01 * Q.topLeftCorner(6, 6);
    Q.bottomRightCorner(6, 6) = 0.01 * Q.bottomRightCorner(6, 6);
    Rt.topLeftCorner(3, 3) = 0.1 * Rt.topLeftCorner(3, 3);
    Rt.bottomRightCorner(6, 6) = 0.1 * Rt.bottomRightCorner(6, 6);
    Rt.bottomRightCorner(1, 1) = 0.1 * Rt.bottomRightCorner(1, 1);
    */
    Q.diagonal()<<0.01,0.01,0.01,
        0.01,0.01,0.01,
        0.1,0.1,0.1,
        0.05,0.05,0.05,
        0.05,0.05,0.05;
    Rt.diagonal()<<0.1,0.1,0.1,
        0.1,0.1,0.1,
        INFINITY,INFINITY,INFINITY; // no optflow for part 1
    cout<<"Q = \n"<<Q<<endl;
    cout<<"Rt = \n"<<Rt<<endl;
    VectorXd initial_state = VectorXd::Zero(15);
    MatrixXd initial_cov = 0.1*MatrixXd::Identity(15,15);
    cout<<"mu0 = \n"<<initial_state<<endl;
    cout<<"sigma0 = \n"<<initial_cov<<endl;

    ekf = new EKF(initial_state,initial_cov,Q,Rt);
    
    ros::spin();
}
