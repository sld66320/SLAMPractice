#include <iostream>
#include <Eigen/Dense>
// #include <Eigen/Cholesky>  
// #include <Eigen/LU>  
// #include <Eigen/QR>  
// #include <Eigen/SVD> 
#include <iostream>
#include <fstream>
#include <ctime>
using namespace std;
using namespace Eigen;

int main()
{
    //读取数据
    ifstream fin("../data2.txt");
   char head[100] = {0};
    fin.getline(head, 100);
    MatrixXf A;
    MatrixXf b;
    A.resize(99, 2);
    b.resize(99, 1);
    cout .precision(4);

//std::cout << "条件数:" << A.norm() * ((A.transpose()*A).inverse()*A.transpose()).norm() << std::endl;
    for(int i = 0; i < 99; i++)
    {
        float x, y;
        fin >> x >> y;
        A.row(i) << x, 1.0f;
        b(i, 0) = y;
    }
     Eigen::JacobiSVD<Eigen::MatrixXf> svd(A);
// //条件数等于最小的奇异值除以最小的奇异值
     double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
     std::cout << "条件数:" << cond << std::endl;
    //真值
    std::cout << "ground truth: 4 1.5" <<  std::endl;
    clock_t startTime,endTime;
    // SVD分解
    startTime = clock();
    Eigen::Matrix<float, 2, 1> svd_result = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    endTime = clock();
    std::cout << "svd result:" <<  svd_result.transpose() << std::endl;
    std::cout << "用时:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

    //qr分解
    startTime = clock();
    Eigen::Matrix<float, 2, 1> qr_result = A.colPivHouseholderQr().solve(b);
    endTime = clock();
    std::cout << "qr result:" <<  qr_result.transpose() << std::endl;
    std::cout << "用时:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;

    // lu
    startTime = clock();
    Eigen::Matrix<float, 2, 1> lu_result = (A.transpose() * A).lu().solve(A.transpose() * b);
    endTime = clock();
    std::cout << "lu result:" <<  lu_result.transpose() << std::endl;
    std::cout << "用时:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    //ldlt
    startTime = clock();
    Eigen::Matrix<float, 2, 1> ldlt_result = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    endTime = clock();
    std::cout << "ldlt result:" <<  ldlt_result.transpose() << std::endl;
    std::cout << "用时:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    //llt
    startTime = clock();
    Eigen::Matrix<float, 2, 1> choloesky_result = (A.transpose() * A).llt().solve(A.transpose() * b);
    endTime = clock();
    std::cout << "choloesky result:" <<  choloesky_result.transpose() << std::endl;
    std::cout << "用时:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
}