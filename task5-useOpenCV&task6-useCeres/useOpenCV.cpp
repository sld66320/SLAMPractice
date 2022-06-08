#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat img1 =  cv::imread("../1.png", 0);
    cv::imshow("原图", img1);
    for(int i = 0; i < img1.rows; i++)
    {
        for(int j = 0; j < img1.cols; j++)
        {
            img1.at<unsigned char>(i, j) = 255 - img1.at<unsigned char>(i, j);
        }
    }
    cv::imshow("取反", img1);
    cv::imwrite("../img1_inv.png", img1);


    cv::waitKey(0);
}