#include "ceres/local_parameterization.h"
#include <Eigen/Geometry>
class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}
    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;
    bool ComputeJacobian(const double* x, double* jacobian) const override;
    int GlobalSize() const override { return 6; }
    int LocalSize() const override { return 6; }
};
bool SE3Parameterization::Plus(const double* x,
                                      const double* delta,
                                      double* x_plus_delta) const {
    for(int i = 3; i < 6; i++)
        x_plus_delta[i] = x[i] + delta[i];

    // x delta换乘矩阵表示
    Eigen::Vector3d temp;
    temp << x[0], x[1], x[2];
    double norm = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    if(abs(norm)>0.1/180*3.1415926)
        temp = temp/norm;
    Eigen::AngleAxisd so3_x(norm, temp);

    temp << delta[0], delta[1], delta[2];
    norm = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    if(abs(norm)>0.1/180*3.1415926)
        temp = temp/norm;
    Eigen::AngleAxisd so3_delta(norm, temp);

    //x左乘R
    Eigen::AngleAxisd so3_x_plus_delta(so3_delta.toRotationMatrix()*so3_x.toRotationMatrix());

    //左乘之后的结果转为李代数
    x_plus_delta[0] = so3_x_plus_delta.axis()(0,0)*so3_x_plus_delta.angle();
    x_plus_delta[1] = so3_x_plus_delta.axis()(1,0)*so3_x_plus_delta.angle();
    x_plus_delta[2] = so3_x_plus_delta.axis()(2,0)*so3_x_plus_delta.angle();

    return true;
}

bool SE3Parameterization::ComputeJacobian(const double* x, double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
    J = Eigen::MatrixXd::Identity(6, 6);
    return true;
}