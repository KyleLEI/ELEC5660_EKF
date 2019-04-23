#include <Eigen/Eigen>

class EKF{
public:
    EKF(const Eigen::VectorXd initial_mean, const Eigen::MatrixXd initial_cov,
        const Eigen::MatrixXd Q_in, const Eigen::MatrixXd R_in);
    Eigen::VectorXd predict(const Eigen::VectorXd u_t, const double dt); 
    void update();

    //Eigen::MatrixXd getCovarianceMatrix() const{return sigma;}
    //Eigen::VectorXd getMean() const{} 

private:
    /* Private Function */
    Eigen::VectorXd f0();    // returns f(mu_(t-1), u_t, 0)
    Eigen::MatrixXd A();
    Eigen::MatrixXd F(const double dt);
    Eigen::MatrixXd U();
    Eigen::MatrixXd V(const double dt);
    Eigen::MatrixXd C();
    Eigen::MatrixXd W();
    Eigen::MatrixXd K();

    /* Private Data */
    Eigen::MatrixXd sigma;  // covariance matrix
    Eigen::VectorXd mu;     // mean
    Eigen::MatrixXd sigma_hat;
    Eigen::VectorXd mu_hat;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    /* Mean of State, mu */
    double x,y,z;           // X1: position p,      PNP
    double roll,pitch,yaw;  // X2: orientation q,   PNP
    double vx,vy,vz;        // X3: linear vel p_dot,OPTFLOW
    double bgx,bgy,bgz;     // X4: gyro bias b_g
    double bax,bay,baz;     // X5: acc bias b_a
    void updateMean();      // Always call this after updating mu

    /* Input u */
    double wx,wy,wz;        // gyro
    double ax,ay,az;        // acc
};
