#include <Eigen/Eigen>

class EKF{
public:
    EKF();
private:
    Eigen::MatrixXd sigma; // covariance matrix
    Eigen::VectorXd mu; // mean

    double x,y,z;           // X1: position p,      PNP
    double roll,pitch,yaw;  // X2: orientation q,   PNP
    double vx,vy,vz;        // X3: linear vel p_dot,OPTFLOW
    double bgx,bgy,bgz;     // X4: gyro bias b_g
    double bax,bay,baz;     // X5: acc bias b_a
}
