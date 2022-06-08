#include<Eigen/Core>
#include<iostream>
using namespace Eigen;
using namespace std;

int main()
{
    //块操作
    	MatrixXf m1(4,4);
	m1<<	1,2,3,4,
		5,6,7,8,
		9,10,11,12,
		13,14,15,16;
	cout<<"Block in the middle"<<endl;
	cout<<m1.block<2,2>(1,1)<<endl<<endl;        //以第二行第二列为起点选一个2*2的矩阵
	for(int i = 1;i <= 3;++i)
	{ //以第二行第二列为起点选一个i*i的矩阵
		cout<<"Block of size "<<i<<"x"<<i<<endl;
		cout<<m1.block(0,0,i,i)<<endl<<endl;
	}

    //块操作可以用来赋值
    	Array22f m2;
	m2<<	1,2, 3,4;
	Array44f m3 = Array44f::Constant(0.6);
	cout<<"Here is the array a:"<<endl<<m3<<endl<<endl;
	m3.block<2,2>(1,1) = m2;
	cout<<"Here is now a with m copied into its central 2x2 block:"<<endl<<m3<<endl<<endl;
	m3.block(0,0,2,3) = m3.block(2,1,2,3);
	cout<<"Here is now a with bottom-right 2x3 block copied into top-left 2x2 block:"<<endl<<m3<<endl<<endl;

	// 行操作
		MatrixXf m4(3,3);
	m4<<	1,2,3,
		4,5,6,
		7,8,9;
	cout<<"Here is the matrix m:"<<endl<<m4<<endl;
	cout<<"2nd Row:"<<m4.row(1)<<endl;
	m4.col(2) += 3*m4.col(0);
	cout<<"After adding 3 times the first column into third column,the matrix m is:\n";

		ArrayXf v(6);
	v<<1,2,3,4,5,6;
	cout<<"v.				(3)="<<endl<<v.head(3)<<endl<<endl;
	cout<<"v.tail<3>()="<<endl<<v.tail<3>()<<endl<<endl;
	v.segment(1,4) *= 2;
	cout<<"after 'v.segment(1,4) *= 2',v="<<endl<<v<<endl;
	// MatrixXf m5(9,1);
	// 	m5<<	1,2,3,
	// 	4,5,6,
	// 	7,8,9;
	// cout << "m5.tail<4>()" << m5.tail<4>() << endl;
}