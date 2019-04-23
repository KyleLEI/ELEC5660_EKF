#include "EKF.h"
#include <cmath>

using namespace Eigen;

EKF::EKF(const VectorXd is){
    sigma = MatrixXd::Zero(21,21);
    mu = VectorXd::Zero(21);
    x = is[0];
    y = is[1];
    z = is[2];
    roll = is[3];
    pitch = is[4];
    yaw = is[5];
    vx = is[6];
    vy = is[7];
    vz = is[8];
    bgx = is[9];
    bgy = is[10];
    bgz = is[11];
    bax = is[12];
    bay = is[13];
    baz = is[14];

    mu.topRows(15) = is;
    sigma.diagonal()<<1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
}

VectorXd EKF::A(){
    /* Linearized about x = mu_(t-1), u = u_t, Q = 0 */
    /* Everything haven't been updated yet, so x = mu_(t-1) */
    /*** Assuming u has been updated before calling this func ***/
    VectorXd A = VectorXd::Zero(15);
    A[4] = - (cos(pitch)*sin(roll)*(bgx + ngx - wx))/cos(roll) - (sin(pitch)*sin(roll)*(bgz + ngz - wz))/cos(roll);
    return A;
}

VectorXd EKF::U(){
    VectorXd U(15);
    U<<0,0,0,-cos(pitch),-1,-cos(pitch)/cos(roll),sin(pitch)*sin(roll)*sin(yaw) - cos(pitch)*cos(yaw),-cos(roll)*cos(yaw),-cos(pitch)*cos(roll),1,1,1,1,1,1;
    return U;
}


