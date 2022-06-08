#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <iostream>
#include<opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include "SnavelyReprojectionError.h"

#include "se3Parameterization.h"
#include "reprojectErrJoc.h"

#include <fstream>
using namespace cv;

//Initial                          6.184818e+02
//Final                            3.577516e+02
//Change                           2.607302e+02

int main()
{

    cv::Mat img1 =  cv::imread("../1.png", 0);
    cv::Mat img2 =  cv::imread("../2.png", 0);
    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05,0 );
    
    //remap去畸变
    cv::Mat map1, map2;
    double alpha = 0;
    //根据缩放比例和裁减情况确定新的内参矩阵
    
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, img1.size(), alpha,  img1.size(), 0);
    //计算映射
    cv::initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix,  img1.size(), CV_16SC2, map1, map2);
    //
    cv::Mat undistortImage1;
    cv::remap(img1, undistortImage1, map1, map2, cv::INTER_LINEAR);

    cv::Mat undistortImage2;
    cv::remap(img2, undistortImage2, map1, map2, cv::INTER_LINEAR);

    //orb特征检测
    std::vector<cv::KeyPoint> keypoint01,keypoint02;//定义两个容器存放特征点
    Ptr<ORB> orb = ORB::create();
    orb->detect(undistortImage1, keypoint01);
	orb->detect(undistortImage2, keypoint02);
    //在两幅图中画出检测到的特征点
    cv::Mat outImg01;
    cv::Mat outImg02;
    cv::drawKeypoints(undistortImage1,keypoint01,outImg01);
    cv::drawKeypoints(undistortImage2,keypoint02,outImg02);
    //cv::imshow("特征点图01",outImg01);
    //cv::imshow("特征点图02",outImg02);
    //waitKey(0);
    Mat descriptors1, descriptors2;
    // //提取特征点的特征向量（128维）
    orb->compute(undistortImage1, keypoint01, descriptors1);
	orb->compute(undistortImage2, keypoint02, descriptors2);

    //特征匹配
    std::vector<DMatch> matches;
    
	//加载汉明距离
	BFMatcher matcher(NORM_HAMMING);
	//两幅图BRIEF匹配
	matcher.match(descriptors1, descriptors2, matches);
    Mat imgMatch;
	drawMatches(undistortImage1, keypoint01, undistortImage2, keypoint02, matches, imgMatch);
	//imshow("所有匹配点对", imgMatch);

    //RANSAC
    std::vector<Point2f>p01,p02;
    std::vector<KeyPoint>goodKeypoint01,goodKeypoint02;
    std::vector<DMatch> goodMatches;
    std::vector<int> point2matchMap;
    Mat imgGoodMatch;
    std::vector<unsigned char> RansacStatus;
    for (size_t i=0;i<matches.size();i++)   
    {
        p01.push_back(keypoint01[matches[i].queryIdx].pt);
        p02.push_back(keypoint02[matches[i].trainIdx].pt);
        point2matchMap.push_back(i);
    }
    Mat EssentialMat= findEssentialMat(p01,p02,NewCameraMatrix,FM_RANSAC,  0.5, 1, RansacStatus);
    int Index = 0;
    for(int i = 0; i < p01.size(); i++)
    {
         if (RansacStatus[i]!=0)
        {
            goodKeypoint01.push_back(keypoint01[matches[point2matchMap[i]].queryIdx]);
            goodKeypoint02.push_back(keypoint02[matches[point2matchMap[i]].trainIdx]);
            DMatch tMatch;
            tMatch.queryIdx = Index;
            tMatch.trainIdx = Index++;
            tMatch.imgIdx = 0;
            goodMatches.push_back(tMatch);
        }
    }
	drawMatches(undistortImage1, goodKeypoint01, undistortImage2, goodKeypoint02, goodMatches, imgGoodMatch);
	//imshow("RANSAC匹配点对", imgGoodMatch);

    Eigen::MatrixXd E(3, 3);
    cv::cv2eigen(EssentialMat, E);
    std::cout << "E = " << E << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d S = U.transpose()* E * V;
    std::cout << "S = " << std::endl << S << std::endl;
    //重新确定奇异值
    float s1 = (S(1,1) + S(2,2))/2;
    S << s1, 0, 0, 0, s1, 0, 0, 0, 0;

    //计算R1 R2 t1 t2
    std::vector<Eigen::Vector3d> vect;
    std::vector<Eigen::Matrix3d> vecR;
    Eigen::AngleAxisd rotZ(M_PI/2, Eigen::Vector3d(0, 0, 1));
    Eigen::AngleAxisd rotZ_(-M_PI/2, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d t1_M = U*rotZ.matrix()*S*U.transpose();
    Eigen::Matrix3d t2_M = U*rotZ_.matrix()*S*U.transpose();
    Eigen::Matrix3d R1_M = U*rotZ.matrix()*V.transpose();
    Eigen::Matrix3d R2_M = U*rotZ_.matrix()*V.transpose();
    Eigen::Vector3d t1(t1_M(2,1), - t1_M(2,0), t1_M(1,0));
    Eigen::Vector3d t2(t2_M(2,1), - t2_M(2,0), t2_M(1,0));
    vect.push_back(t1);
    vect.push_back(t2);
    vecR.push_back(R1_M);
    vecR.push_back(R2_M);
    std::cout << "t1 = " << std::endl << t1 << std::endl;
    std::cout << "t2 = " << std::endl << t2 << std::endl;
    std::cout << "R1 = " << std::endl << R1_M << std::endl;
    std::cout << "R2 = " << std::endl << R2_M << std::endl;

    // 计算归一化坐标
    std::vector<cv::Point2f> vecpt1, vecpt2;
    double fx = K.at<double>(0, 0);
    double cx = K.at<double>(0, 2);
    double fy = K.at<double>(1, 1);
    double cy = K.at<double>(1, 2);
    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        float x1 = (goodKeypoint01[i].pt.x - cx)/fx;
        float y1 = (goodKeypoint01[i].pt.y - cy)/fy;
        float x2 = (goodKeypoint02[i].pt.x - cx)/fx;
        float y2 = (goodKeypoint02[i].pt.y - cy)/fy;
        vecpt1.push_back(cv::Point2f(x1, y1));
        vecpt2.push_back(cv::Point2f(x2, y2));
    }

    // 三角测量
    int minNegNums = 999999;
    Eigen::Matrix3d RGood;
    Eigen::Vector3d tGood;
    cv::Mat points3d;
    // 对恢复的四组可能的解分别进行计算，并依据深度值选出最合理的一个
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            cv::Mat T1 = (cv::Mat_<float>(3,4) << 
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);

            //Eigen::MatrixXd 转 Mat
            cv::Mat T2 = cv::Mat::zeros(Size(4, 3), CV_32F);
            Mat tempM;
            cv::eigen2cv(vecR[i], tempM);
            tempM.copyTo(T2(cv::Rect(0,0,3, 3)));       //这里转换的时候 Eigen矩阵的类型必须是double 被赋值的OpenCV类型可以可以是其他的
            cv::eigen2cv( vect[j], tempM);
            tempM.copyTo(T2(cv::Rect(3,0,1, 3)) );

            // OpenCV三角测量函数
            cv::Mat points_4d;
            cv::triangulatePoints(T1, T2, vecpt1, vecpt2, points_4d);
            cv::Mat T2_ = cv::Mat::eye(Size(4, 4), CV_32F);
            T2.copyTo(T2_(cv::Rect(0, 0, 4, 3)));
            Mat points_4d_2 = T2_*points_4d;        //这里乘法 两个Mat类型必须相同， double不能乘float
            //根据深度值选出一个最优的R和t
            int negNums = 0;
            for(int i = 0; i < vecpt1.size(); i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    points_4d.at<float>(j, i) /= points_4d.at<float>(3, i);
                    points_4d_2.at<float>(j, i) /= points_4d_2.at<float>(3, i);
                }
                if( points_4d.at<float>(2, i)< 0)
                    negNums++;
                if(points_4d_2.at<float>(2, i)< 0)
                    negNums++;
            }
            std::cout << "negnums = " << negNums << std::endl;
            if(negNums < minNegNums)
            {
                minNegNums = negNums;
                RGood = vecR[i];
                tGood = vect[j];
                points3d = points_4d.clone();
            }
        }
    }
    
//    std:: cout << "minNegNums = " << minNegNums << std::endl;
//    std::cout << "R = " << std::endl << RGood << std::endl;             //R21
//    std::cout << "t = " << std::endl << tGood << std::endl;             //t21
//    std::cout << "p = " << std::endl;
//    for(int i = 0; i < points3d.cols; i++)
//    {
//        std::cout << points3d.at<float>(0, i) << ", " << points3d.at<float>(1, i) << ", " << points3d.at<float>(2, i) << ", " << points3d.at<float>(3, i) << std::endl;
//    }

    //BA优化
    //goodKeypoint01 为图1中的特征点
    //goodKeypoint02 为图2中的特征点
    //points3d 为特征点对应的三维坐标
    std::cout << "start ba" << std::endl;
    Eigen::Matrix3d RMc1tow = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d RMc2tow = RGood;
    Eigen::AngleAxisd RVc1tow(RMc1tow);
    Eigen::AngleAxisd RVc2tow(RMc2tow);
    double *pointsCeres = new double[3*points3d.cols];
    for(int i = 0; i < points3d.cols; i++)
    {
        pointsCeres[3*i] = points3d.at<float>(0, i);
        pointsCeres[3*i + 1] = points3d.at<float>(1, i);
        pointsCeres[3*i + 2] = points3d.at<float>(2, i);
    }
    std::cout << RVc1tow.axis()[0] << ", " << RVc1tow.axis()[1] << ", " << RVc1tow.axis()[2] << std::endl;
    std::cout << RVc2tow.axis()[0] << ", " << RVc2tow.axis()[1] << ", " << RVc2tow.axis()[2] << std::endl;

    double camera1[6] = {RVc1tow.axis()[0]*RVc1tow.angle(), RVc1tow.axis()[1]*RVc1tow.angle(), RVc1tow.axis()[2]*RVc1tow.angle(), 0, 0, 0};
    double camera2[6] = {RVc2tow.axis()[0]*RVc2tow.angle(), RVc2tow.axis()[1]*RVc2tow.angle(), RVc2tow.axis()[2]*RVc2tow.angle(), tGood[0], tGood[1], tGood[2]};
    ceres::Problem problem;
    for(int i = 0; i < points3d.cols; i++)
    {

        ceres::CostFunction *cost_function1;
        ceres::CostFunction *cost_function2;
        //自定义雅可比
        cost_function1 = new ReprojectionErrorJoc(fx, fy, cx, cy, goodKeypoint01[i].pt.x, goodKeypoint01[i].pt.y);
        cost_function2 = new ReprojectionErrorJoc(fx, fy, cx, cy, goodKeypoint02[i].pt.x, goodKeypoint02[i].pt.y);

        //自动求导
//        cost_function1 = SnavelyReprojectionError::Create((double)goodKeypoint01[i].pt.x, (double)goodKeypoint01[i].pt.y,  fx, fy, cx, cy);
//        cost_function2 = SnavelyReprojectionError::Create((double)goodKeypoint02[i].pt.x, (double)goodKeypoint02[i].pt.y,  fx, fy, cx, cy);

        //loss function.
        ceres::LossFunction *loss_function1 = new ceres::HuberLoss(1.0);
        ceres::LossFunction *loss_function2 = new ceres::HuberLoss(1.0);
 
        problem.AddResidualBlock(cost_function1, loss_function1, camera1, pointsCeres + 3*i);
        problem.AddResidualBlock(cost_function2, loss_function2, camera2, pointsCeres + 3*i);
    }
    //自定义雅可比的更新方法
    problem.AddParameterBlock(camera1, 6, new SE3Parameterization());
    problem.AddParameterBlock(camera2, 6, new SE3Parameterization());
    std::cout << "Solving ceres BA ... " << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // 计算重投影误差
    std::vector<cv::Point2f> reprojPoints01(goodKeypoint01.size()), reprojPoints02(goodKeypoint01.size()), residuals01(goodKeypoint01.size()), residuals02(goodKeypoint01.size());
    std::vector<double> reprojErr01(goodKeypoint01.size()), reprojErr02(goodKeypoint01.size());

    // 处理相机外参
    Eigen::Vector3d temp;
    temp << camera1[0], camera1[1], camera1[2];
    double norm = sqrt(camera1[0]*camera1[0] + camera1[1]*camera1[1] + camera1[2]*camera1[2]);
    if(abs(norm)>0.1/180*3.1415926)
        temp = temp/norm;
    Eigen::AngleAxisd so3_camera1(norm, temp);
    Eigen::Vector3d t_camera1;
    t_camera1 << camera1[3], camera1[4], camera1[5];

    temp << camera2[0], camera2[1], camera2[2];
    norm = sqrt(camera2[0]*camera2[0] + camera2[1]*camera2[1] + camera2[2]*camera2[2]);
    if(abs(norm)>0.1/180*3.1415926)
        temp = temp/norm;
    Eigen::AngleAxisd so3_camera2(norm, temp);
    Eigen::Vector3d t_camera2;
    t_camera2 << camera2[3], camera2[4], camera2[5];

    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        // 重投影计算
        // 转到相机坐标系
        Eigen::Vector3d p3d;
        p3d << pointsCeres[3*i], pointsCeres[3*i + 1], pointsCeres[3*i + 2];
        Eigen::Vector3d p3dCam1 = so3_camera1*p3d + t_camera1;
        Eigen::Vector3d p3dCam2 = so3_camera2*p3d + t_camera2;

        //投影
        double u1 = fx*p3dCam1(0,0)/p3dCam1(2,0)+cx;
        double v1 = fy*p3dCam1(1,0)/p3dCam1(2,0)+cy;
        double u2 = fx*p3dCam2(0,0)/p3dCam2(2,0)+cx;
        double v2 = fy*p3dCam2(1,0)/p3dCam2(2,0)+cy;

        //保存结果
        reprojPoints01[i] = cv::Point2f(u1, v1);
        reprojPoints02[i] = cv::Point2f(u2, v2);

        cv::Point2f residuals1 = cv::Point2f(u1, v1) - goodKeypoint01[i].pt;
        cv::Point2f residuals2 = cv::Point2f(u2, v2) - goodKeypoint02[i].pt;
        residuals01[i] = residuals1;
        residuals02[i] = residuals2;

        reprojErr01[i] = sqrt(residuals1.x*residuals1.x + residuals1.y*residuals1.y);
        reprojErr02[i] = sqrt(residuals2.x*residuals2.x + residuals2.y*residuals2.y);
    }

    double errSum = 0;
    cv::Point2f residualsAvg(0, 0);
    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        errSum += reprojErr01[i];
        errSum += reprojErr02[i];
    }

    double errAvg = errSum/2/goodKeypoint01.size();
    double errD = 0;
    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        errD += (errAvg-reprojErr01[i])*(errAvg-reprojErr01[i]);
        errD += (errAvg-reprojErr02[i])*(errAvg-reprojErr02[i]);
    }
    errD = errD/2/goodKeypoint01.size();
    double errStd = sqrt(errD);
    std::cout << "平均重投影误差：" << errAvg << std::endl;
    std::cout << "方差：" << errD << std::endl;
    std::cout << "标准差：" << errStd << std::endl;

    //可视化投影点和提取点
    cv::Mat showReprojImg1 = undistortImage1.clone();
    cv::Mat showReprojImg2 = undistortImage2.clone();
    cvtColor(showReprojImg1, showReprojImg1, CV_GRAY2BGR);
    cvtColor(showReprojImg2, showReprojImg2, CV_GRAY2BGR);
    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        cv::circle(showReprojImg1, goodKeypoint01[i].pt, 2, cv::Scalar(255, 0, 0));
        cv::circle(showReprojImg2, goodKeypoint02[i].pt, 2, cv::Scalar(255, 0, 0));

        cv::circle(showReprojImg1, reprojPoints01[i], 2, cv::Scalar(0, 0, 255));
        cv::circle(showReprojImg2, reprojPoints02[i], 2, cv::Scalar(0, 0, 255));
    }

    cv::imshow("图1重投影示意图", showReprojImg1);
    cv::imshow("图2重投影示意图", showReprojImg2);
    cv::waitKey(0);

    //写入重投影误差到文件
    std::ofstream fout1("../residuals01.txt");
    std::ofstream fout2("../residuals02.txt");
    for(int i = 0; i < goodKeypoint01.size(); i++)
    {
        fout1 << residuals01[i].x << " " << residuals01[i].y << std::endl;
        fout2 << residuals02[i].x << " " << residuals02[i].y << std::endl;
    }
    fout1.close();
    fout2.close();
    // 梯度检查
//    ceres::CostFunction* my_cost_function = new ReprojectionErrorJoc(fx, fy, cx, cy, goodKeypoint02[0].pt.x, goodKeypoint02[0].pt.y);
//    ceres::LocalParameterization* my_parameterization = new SE3Parameterization();
//    ceres::NumericDiffOptions numeric_diff_options;
//    std::vector<const ceres::LocalParameterization* > local_parameterizations;
//    local_parameterizations.push_back(my_parameterization);
//    local_parameterizations.push_back(NULL);
//
//    double parameter1[6] = {camera2[0], camera2[1], camera2[2], camera2[3], camera2[4], camera2[5]};
//    double parameter2[3] = {pointsCeres[0], pointsCeres[1], pointsCeres[2]};
//    std::vector<double*> parameter_blocks;
//    parameter_blocks.push_back(parameter1);
//    parameter_blocks.push_back(parameter2);
//
//    ceres::GradientChecker gradient_checker(my_cost_function, &local_parameterizations, numeric_diff_options);
//    ceres::GradientChecker::ProbeResults results;
//    if(!gradient_checker.Probe(parameter_blocks.data(), 0.1, &results))
//    {
//        LOG(ERROR) << "An error has occurred:\n" << results.error_log;
//    }


}