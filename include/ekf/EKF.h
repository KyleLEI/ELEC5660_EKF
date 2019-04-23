#include <Eigen/Eigen>

class EKF{
public:
    EKF(const Eigen::VectorXd initial_state);
    Eigen::VectorXd predict(Eigen::VectorXd u_t);
    void update();

    //Eigen::MatrixXd getCovarianceMatrix() const{return sigma;}
    //Eigen::VectorXd getMean() const{} 

private:
    /* Private Function */
    Eigen::VectorXd f();
    Eigen::VectorXd A();
    //Eigen::VectorXd B();
    Eigen::VectorXd U();
    Eigen::MatrixXd C();
    Eigen::MatrixXd W();
    Eigen::MatrixXd K();

    /* Private Data */
    Eigen::MatrixXd sigma; // covariance matrix, (5*3+2*3), PNP
    Eigen::VectorXd mu; // mean, (5*3+2*3), PNP

    /* State X */
    double x,y,z;           // X1: position p,      PNP
    double roll,pitch,yaw;  // X2: orientation q,   PNP
    double vx,vy,vz;        // X3: linear vel p_dot,OPTFLOW
    double bgx,bgy,bgz;     // X4: gyro bias b_g
    double bax,bay,baz;     // X5: acc bias b_a

    /* Noise n */
    double ngx,ngy,ngz;     // gyro noise
    double nax,nay,naz;     // acc noise
    double nbgx,nbgy,nbgz;  // gyro bias noise
    double nbax,nbay,nbaz;  // acc bias noise

    /* Input u */
    double wx,wy,wz;        // gyro
    double ax,ay,az;        // acc
};
