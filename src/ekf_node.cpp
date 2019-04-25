#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>

#include "EKF.h"

EKF* ekf=nullptr;
using namespace std;
using namespace Eigen;
ros::Publisher odom_pub;
MatrixXd Q = MatrixXd::Identity(15, 15);
//MatrixXd Rt = MatrixXd::Identity(6,6);
MatrixXd Rt = MatrixXd::Identity(3,3);
double imu_time, last_imu_time=-1.0; 

void imu_callback(const sensor_msgs::Imu::ConstPtr &msg)
{
    if(ekf==nullptr)
        return;
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
    ekf->predict(u,imu_time-last_imu_time);
    last_imu_time = imu_time;
}

void publish_odom(VectorXd m){
    double x_m=m(0),y_m=m(1),z_m=m(2),roll_m=m(3),pitch_m=m(4),yaw_m=m(5);
    Vector3d rpy_m{roll_m,pitch_m,yaw_m};
    Matrix3d R_m = EKF::util_RPYToRot(rpy_m);
    Quaterniond Q_ekf(R_m);

    nav_msgs::Odometry odom_ekf;
    odom_ekf.header.frame_id = "world";
    odom_ekf.pose.pose.position.x = x_m;
    odom_ekf.pose.pose.position.y = y_m;
    odom_ekf.pose.pose.position.z = z_m;
    odom_ekf.pose.pose.orientation.w = Q_ekf.w();
    odom_ekf.pose.pose.orientation.x = Q_ekf.x();
    odom_ekf.pose.pose.orientation.y = Q_ekf.y();
    odom_ekf.pose.pose.orientation.z = Q_ekf.z();
    odom_pub.publish(odom_ekf);
}

//Rotation from the camera frame to the IMU frame
Vector3d T_ic;
Eigen::Matrix3d R_ic;
void pnp_callback(const nav_msgs::Odometry::ConstPtr &msg)
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
    
    VectorXd z(6);

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

    Vector3d T_wc = -1*T_cw;
    Matrix3d R_wc = R_cw.transpose();
    Vector3d T_ci = -1*T_ic;
    Matrix3d R_ci = R_ic.transpose();
   
    Matrix3d R_wi = R_wc*R_ci;
    Vector3d T_wi = R_wc*T_ci+T_wc;
    
    Vector3d rpy_pnp = EKF::util_RotToRPY(R_wi);
    //cout<<"Z rpy = \n"<<rpy_pnp/M_PI*180<<endl;
    //cout<<"Z trans = \n"<<T_wi<<endl;
    z<< T_wi,rpy_pnp;
    if(ekf==nullptr){// not initialized
        VectorXd initial_state(15);
        initial_state<<z,VectorXd::Zero(9);
        MatrixXd initial_cov = MatrixXd::Zero(15,15);
        initial_cov.diagonal()<<
            0.01,0.01,0.01,
            0.01,0.01,0.01,
            0.05,0.05,0.05,
            0.1,0.1,0.1,
            0.1,0.1,0.1;

        ekf = new EKF(initial_state,initial_cov,Q,Rt);
        return;
    }
    ekf->update1(z);
    std::cout<<"sigma =\n"<<ekf->getCovariance().diagonal()<<std::endl;
    std::cout<<"mu =\n"<<ekf->getMean()<<std::endl<<std::endl;

    //VectorXd m = ekf->getMean();
    publish_odom(ekf->getMean());
}

void optflow_callback(const nav_msgs::Odometry::ConstPtr &msg){
    double  of_vx = msg->twist.twist.linear.x,
            of_vy = msg->twist.twist.linear.y,
            of_z = msg->pose.pose.position.z;
    if(ekf==nullptr){
        VectorXd initial_state = VectorXd::Zero(15);
        initial_state(2) = of_z;
        MatrixXd initial_cov = MatrixXd::Zero(15,15);
        initial_cov.diagonal()<<
            0.01,0.01,0.01,
            0.01,0.01,0.01,
            0.05,0.05,0.05,
            0.1,0.1,0.1,
            0.1,0.1,0.1;

        ekf = new EKF(initial_state,initial_cov,Q,Rt);
        return;
    }

    Vector3d z{of_vx,of_vy,of_z};
    ekf->update2(z);
    std::cout<<"sigma =\n"<<ekf->getCovariance().diagonal()<<std::endl;
    std::cout<<"mu =\n"<<ekf->getMean()<<std::endl<<std::endl;
    publish_odom(ekf->getMean());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ekf");
    ros::NodeHandle n("~");
    ros::Subscriber s1 = n.subscribe("imu", 1000, imu_callback);
    //ros::Subscriber s2 = n.subscribe("tag_odom", 1000, pnp_callback);
    ros::Subscriber s3 = n.subscribe("optflow_odom",1000,optflow_callback);
    odom_pub = n.advertise<nav_msgs::Odometry>("ekf_odom", 100);
    R_ic = Quaterniond(0, 0, 0, 1).toRotationMatrix();
    T_ic<<0.1, 0, 0.03;
    // Q imu covariance matrix; Rt visual odomtry covariance matrix
    // You should also tune these parameters
    /*
    Q.topLeftCorner(6, 6) = 0.01 * Q.topLeftCorner(6, 6);
    Q.bottomRightCorner(6, 6) = 0.01 * Q.bottomRightCorner(6, 6);
    Rt.topLeftCorner(3, 3) = 0.1 * Rt.topLeftCorner(3, 3);
    Rt.bottomRightCorner(6, 6) = 0.1 * Rt.bottomRightCorner(6, 6);
    Rt.bottomRightCorner(1, 1) = 0.1 * Rt.bottomRightCorner(1, 1);
    */
    Q.diagonal()<<
        0.01,0.01,0.01, // no effect
        0.01,0.01,0.01, // gyro
        0.01,0.01,0.01, // acc
        0.05,0.05,0.05,    // gyro bias
        0.05,0.05,0.05;    // acc bias
    //Rt.diagonal()<<
    //    0.01,0.01,0.01, // PnP position
    //    0.01,0.01,0.01; // PnP orientation
    Rt.diagonal()<<
        0.1,0.1,0.005; // optflow vx vy h
        
    ros::spin();
}
