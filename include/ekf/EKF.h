#include <Eigen/Eigen>
#include <cmath>

class EKF{
public:
    EKF(const Eigen::VectorXd initial_mean, const Eigen::MatrixXd initial_cov,
        const Eigen::MatrixXd Q_in, const Eigen::MatrixXd R_pnp, const Eigen::MatrixXd R_optflow);
    Eigen::VectorXd predict(const Eigen::VectorXd u_t, const double dt); // IMU
    void update1(const Eigen::VectorXd z_t); // PnP
    void update2(const Eigen::Vector3d z_t); // optflow

    Eigen::MatrixXd getCovariance() const{return sigma;}
    Eigen::VectorXd getMean() const{return mu;} 

    inline static Eigen::Vector3d util_RotToRPY(Eigen::Matrix3d R){
        Eigen::Vector3d rpy;

        double roll = asin(R(2,1));
        double yaw = atan2(-R(0,1)/cos(roll),R(1,1)/cos(roll));
        double pitch = atan2(-R(2,0)/cos(roll),R(2,2)/cos(roll));
        rpy<<roll,pitch,yaw;

        return rpy;
    }
    inline static Eigen::Matrix3d util_RPYToRot(Eigen::Vector3d rpy){
        Eigen::Matrix3d Rot;
        double roll=rpy(0),pitch=rpy(1),yaw=rpy(2);

        Rot<<cos(pitch)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw),
            -cos(roll)*sin(yaw),
            cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw),
            cos(pitch)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll),
            cos(roll)*cos(yaw),
            sin(pitch)*sin(yaw) - cos(pitch)*cos(yaw)*sin(roll),
            -cos(roll)*sin(pitch),
            sin(roll),
            cos(pitch)*cos(roll);

        return Rot;
    }
    inline static double util_EulerRange(double in){
        if(in>=M_PI)
            in -= 2*M_PI;
        else if (in<-M_PI)
            in += 2*M_PI;
        return in;
    }

private:
    /* Private Function */
    /* Process Model */
    Eigen::VectorXd f0();    // returns f(mu_(t-1), u_t, 0)
    Eigen::MatrixXd A();
    Eigen::MatrixXd F(const double dt);
    Eigen::MatrixXd U();
    Eigen::MatrixXd V(const double dt);
    /* Measurement Model */
    Eigen::MatrixXd C1;
    Eigen::MatrixXd C2();
    Eigen::VectorXd g1();
    Eigen::Vector3d g2();
    /* Kalman Gain */
    Eigen::MatrixXd K(const Eigen::MatrixXd& C, const Eigen::MatrixXd& R);
    //Eigen::MatrixXd K1();
    //Eigen::MatrixXd K2();

    /* Private Data */
    Eigen::MatrixXd sigma;  // covariance matrix
    Eigen::VectorXd mu;     // mean
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R_pnp;
    Eigen::MatrixXd R_optflow;

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
