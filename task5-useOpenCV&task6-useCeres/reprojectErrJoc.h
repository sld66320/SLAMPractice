#include "ceres/sized_cost_function.h"
#include <Eigen/Geometry>

class ReprojectionErrorJoc: public ceres::SizedCostFunction<2, 6, 3>
{
public:
    ReprojectionErrorJoc(double fx_,
                            double fy_,
                            double cx_,
                            double cy_,
                            double observed_x_,
                            double observed_y_)
            : fx(fx_), fy(fy_), cx(cx_), cy(cy_),
              observed_x(observed_x_),
              observed_y(observed_y_){}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

private:
    double observed_x;
    double observed_y;
    double fx;
    double fy;
    double cx;
    double cy;
};

bool ReprojectionErrorJoc::Evaluate(const double * const *parameters, double *residuals, double **jacobians) const
{
    // 初始化一些参数
    const double* camera = parameters[0];
    Eigen::Vector3d temp;
    temp << camera[0], camera[1], camera[2];
    double norm = sqrt(camera[0]*camera[0] + camera[1]*camera[1] + camera[2]*camera[2]);

    if(abs(norm)>0.1/180*3.1415926)
        temp = temp/norm;
    Eigen::AngleAxisd so3_camera(norm, temp);
    Eigen::Vector3d t_camera;
    Eigen::Vector3d p3d;
    p3d << parameters[1][0], parameters[1][1], parameters[1][2];
    t_camera << camera[3], camera[4], camera[5];

    // 计算残差
    Eigen::Vector3d p3d_cam = so3_camera.toRotationMatrix()*p3d+t_camera;
    double reProj_u = fx*p3d_cam(0,0)/p3d_cam(2,0) + cx;
    double reProj_v = fy*p3d_cam(1,0)/p3d_cam(2,0) + cy;
    residuals[0] = reProj_u - observed_x;
    residuals[1] = reProj_v - observed_y;

    // 计算雅可比 jacobians 第一行代表参差对相机外参的偏导，共12个参数，第二行代表对三维点的偏导，共6个参数
    // 先计算残差对相机坐标系下p的导数
    Eigen::Matrix<double, 2, 3> residuals2camP;
    residuals2camP << fx/p3d_cam(2,0) , 0, -fx*p3d_cam(0,0)/(p3d_cam(2,0)*p3d_cam(2,0)), 0, fy/p3d_cam(2,0), -fy*p3d_cam(1,0)/(p3d_cam(2,0)*p3d_cam(2,0));
    //相机坐标系下的点P对旋转轴的偏导数
    temp = -(so3_camera.toRotationMatrix()*p3d);
    Eigen::Matrix<double, 3, 3> camP2so3;
    camP2so3 << 0, -temp(2, 0), temp(1, 0), temp(2, 0), 0, -temp(0, 0), -temp(1, 0), temp(0, 0), 0;
    //相机坐标系下的点P对平移向量的偏导数
    Eigen::Matrix<double, 3, 3> camP2t;
    camP2t << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    //相机坐标系下的点P对世界坐标系下的点P
    Eigen::Matrix<double, 3, 3> camP2wordP = so3_camera.toRotationMatrix();

    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.block<2,3>(0,0) = residuals2camP*camP2so3;
            J_se3.block<2,3>(0,3) = residuals2camP*camP2t;
        }
        if(jacobians[1] != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_point(jacobians[1]);
            J_point = residuals2camP * camP2wordP;
        }
    }

    return true;
}