#include<Eigen/Core>
#include<Eigen/Geometry>
#include<iostream>

int main()
{
    // 初始化Imu 到 相机的四元数和平移向量
    Eigen::Quaterniond q_I2cl_JPL(0.99090224973327068, 0.13431639597354814, 0.00095051670014565813, -0.0084222184858180373);
    Eigen::Quaterniond q_I2cr_JPL(0.99073762672679389, 0.13492462817073628, -0.00013648999867379373, -0.015306242884176362);
    Eigen::Vector3d p_I2cl(-0.050720060477640147, -0.0017414170413474165, 0.0022943667597148118);
    Eigen::Vector3d p_I2cr(0.051932496584961352, -0.0011555929083120534, 0.0030949732069645722);
    
    //JPL转ham
    Eigen::Quaterniond q_I2cl_ham = q_I2cl_JPL.inverse();
    Eigen::Quaterniond q_I2cr_ham = q_I2cr_JPL.inverse();

    //构造T
    Eigen::Isometry3d T_I2cl = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_I2cr = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_cl2cr = Eigen::Isometry3d::Identity();
    T_I2cl.rotate(q_I2cl_ham);
    T_I2cr.rotate(q_I2cr_ham);
    T_I2cl.pretranslate(p_I2cl);
    T_I2cr.pretranslate(p_I2cr);
    T_cl2cr = T_I2cl.matrix().inverse()*T_I2cr.matrix();

    //计算四元数和平移向量
    Eigen::Quaterniond q_cl2cr_ham(T_cl2cr.rotation());
    Eigen::Quaterniond q_cl2cr_JPL = q_cl2cr_ham.inverse();
    Eigen::Vector3d p_cI2cr = T_cl2cr.translation();

    std:: cout << "q_cl2cr_JPL = " << q_cl2cr_JPL.coeffs() << "\r\np_cl2cr = " << p_cI2cr << "\r\n";
}