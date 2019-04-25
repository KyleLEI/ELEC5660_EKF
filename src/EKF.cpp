#include "EKF.h"
#include <iostream>
using namespace std;

using namespace Eigen;

EKF::EKF(const VectorXd is, const MatrixXd ic, const MatrixXd Q_in, const MatrixXd R_pnp_in, const MatrixXd R_optflow_in){
    mu = is;
    updateMean();
    sigma = ic;
    Q = Q_in; // 15*15
    R_pnp = R_pnp_in; // 6*6
    R_optflow = R_optflow_in; // 3*3

    /* Initialize the linear measurement matrices */
    C1 = MatrixXd::Identity(6,15);
}

inline void EKF::updateMean(){
    x = mu[0];
    y = mu[1];
    z = mu[2];
   
    /* Deal with Euler angle discontinuity */
    mu[3] = util_EulerRange(mu[3]);
    mu[4] = util_EulerRange(mu[4]);
    mu[5] = util_EulerRange(mu[5]);
    roll = mu[3];
    pitch = mu[4];
    yaw = mu[5];
    vx = mu[6];
    vy = mu[7];
    vz = mu[8];
    bgx = mu[9];
    bgy = mu[10];
    bgz = mu[11];
    bax = mu[12];
    bay = mu[13];
    baz = mu[14];
}

MatrixXd EKF::A(){
    /*** Assuming u has been updated before the call ***/
    MatrixXd A = MatrixXd::Zero(15,15);

    /* df0/droll */
    A(4,3) = (bgz*cos(pitch) - bgx*sin(pitch) - wz*cos(pitch) + wx*sin(pitch))/(cos(roll)*cos(roll));
    A(5,3) = (sin(pitch)*sin(roll)*(bgx - wx))/(cos(roll)*cos(roll)) - (cos(pitch)*sin(roll)*(bgz - wz))/(cos(roll)*cos(roll));
    A(6,3) = sin(roll)*sin(yaw)*(ay - bay) + cos(pitch)*cos(roll)*sin(yaw)*(az - baz) - cos(roll)*sin(pitch)*sin(yaw)*(ax - bax);
    A(7,3) = cos(roll)*cos(yaw)*sin(pitch)*(ax - bax) - cos(pitch)*cos(roll)*cos(yaw)*(az - baz) - cos(yaw)*sin(roll)*(ay - bay);
    A(8,3) = cos(roll)*(ay - bay) - cos(pitch)*sin(roll)*(az - baz) + sin(pitch)*sin(roll)*(ax - bax);

    /* df0/dpitch */
    A(3,4) = sin(pitch)*(bgx - wx) - cos(pitch)*(bgz - wz);
    A(4,4) = - (cos(pitch)*sin(roll)*(bgx - wx))/cos(roll) - (sin(pitch)*sin(roll)*(bgz - wz))/cos(roll);
    A(5,4) = (cos(pitch)*(bgx - wx))/cos(roll) + (sin(pitch)*(bgz - wz))/cos(roll);
    A(6,4) = (az - baz)*(cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) - (ax - bax)*(cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw));
    A(7,4) = (az - baz)*(cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) - (ax - bax)*(sin(pitch)*sin(yaw) - cos(pitch)*cos(yaw)*sin(roll));
    A(8,4) = - cos(pitch)*cos(roll)*(ax - bax) - cos(roll)*sin(pitch)*(az - baz);

    /* df0/dyaw */
    A(6,5) = - (ax - bax)*(cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) - (az - baz)*(sin(pitch)*sin(yaw) - cos(pitch)*cos(yaw)*sin(roll)) - cos(roll)*cos(yaw)*(ay - bay);
    A(7,5) = (ax - bax)*(cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) + (az - baz)*(cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw)) - cos(roll)*sin(yaw)*(ay - bay);

    /* df0/dp_dot */
    A(0,6) = 1.0;
    A(1,7) = 1.0;
    A(2,8) = 1.0;

    /* df0/dbgx */
    A(3,9) = -cos(pitch);
    A(4,9) = -(sin(pitch)*sin(roll))/cos(roll);
    A(5,9) = sin(pitch)/cos(roll);

    /* df0/dbgy */
    A(4,10) = -1;

    /* df0/dbgz */
    A(3,11) = -sin(pitch);
    A(4,11) = (cos(pitch)*sin(roll))/cos(roll);
    A(5,11) = -cos(pitch)/cos(roll);

    /* df0/dbax */
    A(6,12) = sin(pitch)*sin(roll)*sin(yaw) - cos(pitch)*cos(yaw);
    A(7,12) = - cos(pitch)*sin(yaw) - cos(yaw)*sin(pitch)*sin(roll);
    A(8,12) = cos(roll)*sin(pitch);

    /* df0/dbay */
    A(6,13) = cos(roll)*sin(yaw);
    A(7,13) = -cos(roll)*cos(yaw);
    A(8,13) = -sin(roll);

    /* df0/dbaz */
    A(6,14) = - cos(yaw)*sin(pitch) - cos(pitch)*sin(roll)*sin(yaw);
    A(7,14) = cos(pitch)*cos(yaw)*sin(roll) - sin(pitch)*sin(yaw);
    A(8,14) = -cos(pitch)*cos(roll);

    return A;
}

inline MatrixXd EKF::F(const double dt){
    MatrixXd I = MatrixXd::Identity(15,15);
    return I+dt*A();
}

MatrixXd EKF::U(){
    /*** Assuming u has been updated before the call ***/
    MatrixXd U = MatrixXd::Zero(15,15);

    /* df/dngx */
    U(3,3) = -cos(pitch);
    U(4,3) = -(sin(pitch)*sin(roll))/cos(roll);
    U(5,3) = sin(pitch)/cos(roll);

    /* df/dngy */
    U(4,4) = -1.0;

    /* df/dngz */
    U(3,5) = -sin(pitch);
    U(4,5) = (cos(pitch)*sin(roll))/cos(roll);
    U(5,5) = -cos(pitch)/cos(roll);

    /* df/dnax */
    U(6,6) = sin(pitch)*sin(roll)*sin(yaw) - cos(pitch)*cos(yaw);
    U(7,6) = - cos(pitch)*sin(yaw) - cos(yaw)*sin(pitch)*sin(roll);
    U(8,6) = cos(roll)*sin(pitch);

    /* df/dnay */
    U(6,7) = cos(roll)*sin(yaw);
    U(7,7) = -cos(roll)*cos(yaw);
    U(8,7) = -sin(roll);

    /* df/dnaz */
    U(6,8) = - cos(yaw)*sin(pitch) - cos(pitch)*sin(roll)*sin(yaw);
    U(7,8) = cos(pitch)*cos(yaw)*sin(roll) - sin(pitch)*sin(yaw);
    U(8,8) = -cos(pitch)*cos(roll);

    /* df/dnbg, df/dnba */
    U(9,9) = 1.0;
    U(10,10) = 1.0;
    U(11,11) = 1.0;
    U(12,12) = 1.0;
    U(13,13) = 1.0;
    U(14,14) = 1.0;

    return U;
}

inline MatrixXd EKF::V(const double dt){
    return dt*U();
}

VectorXd EKF::f0(){
    /*** Assuming u has been updated before the call ***/
    VectorXd f(15);
    f<<vx,vy,vz,
        - cos(pitch)*(bgx - wx) - sin(pitch)*(bgz - wz),
        wy - bgy + (cos(pitch)*sin(roll)*(bgz - wz))/cos(roll) - (sin(pitch)*sin(roll)*(bgx - wx))/cos(roll),
        (sin(pitch)*(bgx - wx))/cos(roll) - (cos(pitch)*(bgz - wz))/cos(roll),
        (ax - bax)*(cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) + (az - baz)*(cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw)) - cos(roll)*sin(yaw)*(ay - bay),
        (ax - bax)*(cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) + (az - baz)*(sin(pitch)*sin(yaw) - cos(pitch)*cos(yaw)*sin(roll)) + cos(roll)*cos(yaw)*(ay - bay),
        sin(roll)*(ay - bay) + cos(pitch)*cos(roll)*(az - baz) - cos(roll)*sin(pitch)*(ax - bax) - 9.81,
        0,0,0,0,0,0;
    return f;
}

VectorXd EKF::predict(const VectorXd u, const double dt){
    /* Update u */
    wx = u[0];
    wy = u[1];
    wz = u[2];
    ax = u[3];
    ay = u[4];
    az = u[5];
   
    MatrixXd Ft = F(dt);
    MatrixXd Vt = V(dt); 
    /* Estimate mean */
    mu += dt*f0();
    updateMean();

    /* Estimate coveriance */
    sigma = Ft*sigma*(Ft.transpose())+Vt*Q*(Vt.transpose());
    return mu;
}

inline VectorXd EKF::g1(){
    VectorXd g(6);
    g<<x,y,z,roll,pitch,yaw;
    return g;
}

inline MatrixXd EKF::K(const MatrixXd& C, const MatrixXd& R){
    MatrixXd Ct = C;
    MatrixXd Ct_T = Ct.transpose();
    return sigma*Ct_T*((Ct*sigma*Ct_T+R).inverse());
}

void EKF::update1(const VectorXd zt){
    MatrixXd Kt = K(C1,R_pnp);//15*6

    /* Deal with Euler angle discontinuity */
    VectorXd tmp = zt-g1();
    tmp(3)=util_EulerRange(tmp(3));
    tmp(4)=util_EulerRange(tmp(4));
    tmp(5)=util_EulerRange(tmp(5));
    //cout<<"tmp = \n"<<tmp<<endl;

    mu += Kt*tmp;
    updateMean();
    sigma -= Kt*C1*sigma;
}

inline Vector3d EKF::g2(){
    return Vector3d{
        vx*(cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) - vy*cos(roll)*sin(yaw),
        vx*(cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) + vy*cos(roll)*cos(yaw),
        z*cos(pitch)*cos(roll)};
}

MatrixXd EKF::C2(){
    MatrixXd C = MatrixXd::Zero(3,15);
    C(2,2) = cos(pitch)*cos(roll);
    C(2,3) = -z*cos(pitch)*sin(roll);
    C(2,4) = -z*cos(roll)*sin(pitch);

    C(0,3) = sin(yaw)*(vy*sin(roll) - vx*cos(roll)*sin(pitch));
    C(1,3) = -cos(yaw)*(vy*sin(roll) - vx*cos(roll)*sin(pitch));
    C(0,4) = -vx*(cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw));
    C(1,4) = vx*cos(pitch)*cos(yaw)*sin(roll) - vx*sin(pitch)*sin(yaw);
    C(0,5) = - vx*(cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) - vy*cos(roll)*cos(yaw);
    C(1,5) = vx*(cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) - vy*cos(roll)*sin(yaw);
    C(0,6) = cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw);
    C(1,6) = cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll);
    C(0,7) = -cos(roll)*sin(yaw);
    C(1,7) = cos(roll)*cos(yaw);
    return C;
}

void EKF::update2(const Vector3d zt){
    MatrixXd C = C2();// freeze mu
    // zt is in the imu frame
    MatrixXd Kt = K(C,R_optflow);//15*3
    
    mu += Kt*(zt-g2());
    updateMean();
    sigma -= Kt*C*sigma;
}
