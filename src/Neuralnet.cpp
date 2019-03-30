/*
 * net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */


#include <iostream>
#include <armadillo>
#include <algorithm>    // std::random_shuffle
#include <random>


#include "Net.h"
#include "math.h"
#include "dataframe.h"

using namespace std;
using namespace arma;
using namespace vSpace;

inline double relu(const double & d){
	return d > 0 ? d : 0;
}

inline double Lrelu(const double & d){
	return d > 0 ? d : 0.01*d;
}

inline double DLrelu(const double & d){
	return d > 0 ? 1 : 0.01;
}

inline double I(const double & t){
	return t;
}

inline double One(const double & t){
	return 1;
}

inline double Drelu(const double & d){
	return d > 0 ? 1 : 0;
}

inline double sm(const double & t){
	return (t>10)?t:log(1+exp(t));
}
inline double Dsm(const double & t){
	return (t>100)?1:exp(t)/(1+exp(t));
}

inline void grad(const Net & N,Net & G,const double & target){

	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	const vector<double (*)(const double &)> deriv=G.getFs();
//	Deal with end node if net is not empty
	if(n_layers > 0){
//		cout << N.n(n_layers-1,0) << endl;
		G.v(n_layers-1,0) = 2*deriv[n_layers-1](N.n(n_layers-1,0))*( N.v(n_layers-1,0)  -target);
//		cout <<deriv[n_layers-1](N.n(n_layers-1,0))<<endl;
	}

	for(int l = n_layers-2 ; l != -1 ; --l){
//		Coeffs of G store partial diff with respect to that coeff
		for(int start = 0 ; start != layers[l] ; ++start){
			for(int end = 0; end != layers[l+1]; ++end){
				G.c(l,start,end) = N.v(l,start)*G.v(l+1,end);
			}
		}
//		Nodes of G store partial diff with respect to that node value
		for(int start = 0 ; start != layers[l] ; ++start){
			G.v(l,start) = 0;
			for(int end = 0; end != layers[l+1]; ++end){
				G.v(l,start) += N.c(l,start,end)*deriv[l](N.n(l,start))*G.v(l+1,end);
			}
		}
	}
}

static const int nassets{487};
static const int nlines{756};
static const double end_train{11};
static dataframe Data{756,nassets,"cleanIndex.csv"};
static mat Train = Data.getData().rows(0,end_train);

arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & etha,const double & eps){
	mat v{v0};
	mat g = grad(v);
//	double old_n = norm(g);
	double eth = etha;
//	double temp = old_n;
	const int maxIt{100000};
	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{
			v = v-eth*g;
			g = grad(v);
//			cout << "grad" << norm(g) << endl;
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}

arma::mat acc_descent(const arma::mat& v0 , std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
	mat x{v0};
	mat y{v0};
	mat y_old{v0};
	mat g = grad(v0);
	double s{0};
	const int maxIt{100};
	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return x;
		}
		else{
			s = ((double)i)/(i+3.);
			y = x - eth*g;
			x = x*(1+s)-eth*g*(1+s)-s*y_old;
			g = grad(x);
			y_old = y;
//			cout << norm(g)<<endl;
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return x;

}

arma::mat stochastic_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat&,mt19937 &)> & grad,const double & etha,const double & eps){
	mat v{v0};
	const int maxIt{1000};
	mt19937 gen;
	mat g = grad(v,gen);
	double eth = etha;
	double lmin = 0;
	double min = -1 ;
	double lin = 0.1;
	double err = 0;
	vector<double (*)(const double &)> fs{I,Lrelu,Lrelu,Lrelu};
	Net N = Net(vector<int>{487,250,125,1},fs);



	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{

			lmin = 0;
			min = -1 ;
			lin = 0.1;
			for(int j = 0 ; j != 3 ; ++j){
				err = 0;
				N.get_coeffs() = v-eth*lin*g;

				for(int d = 0 ; d != ::end_train+1-10 ; ++d){
				N.n(0,0) = 0;
				for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
					N.n(0,s) = Train(0,s);
				}
				N.update();
				err += pow(N.v(4-1,0)-Train(10,0),2);
				}

				if( min == -1 or min > err){
//					cout << pow(N.v(4-1,0)-Train(10,0),2)<< endl;
					min = err;
					lmin = lin;
				}
				lin *= 10;
			}
			eth = lmin*eth;
			cout << lmin <<endl;
			v = v-eth*g;


			g = grad(v,gen);
//			cout << norm(g) << endl;
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}

int main(){
//	Data processing

//	const int nassets{487};
//	const int nlines{756};
//	const double end_train{10};
//	dataframe Data{756,nassets,"cleanIndex.csv"};
//	mat Train = Data.getData().rows(0,end_train);


//	Define fully connected neural net : layer0 : 500 nodes / layer 1 : 250  nodes / layer 2 : 125 nodes / layer 3 : 1 node

//	For layer 0 we will add 0s because we only have 486 assets.

	vector<double (*)(const double &)> fs{I,Lrelu,Lrelu,Lrelu};
	vector<double (*)(const double &)> ds{One,DLrelu,DLrelu,DLrelu};


	Net N = Net(vector<int>{487,250,125,1},fs);
	Net G = Net(vector<int>{487,250,125,1},ds); /* we do not care of the target we just want the structure*/
	vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);




	std::function<arma::mat(const arma::mat & )> g = [&N,&G,&res_grad](const arma::mat & v){
	N.get_coeffs() = v;
	res_grad.fill(0.);
	for(int d = 0 ; d != ::end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.n(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.n(0,s) = Train(d,s);
		}
		N.update();
		grad(N,G,::Train(d+10,0));
		res_grad += G.get_coeffs();
	}
//	cout << N.v(3,0)<<endl;
	return res_grad/(::end_train+1-10);
	};


//	vector<double (*)(const double &)> fs{I,sm,sm,sm};
//	vector<double (*)(const double &)> ds{I,Dsm,Dsm,Dsm};
//
//
//	Net N = Net(vector<int>{487,500,125,1},fs);
//	Net G = Net(vector<int>{487,500,125,1},ds); /* we do not care of the target we just want the structure*/
//	vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);
//
//
//
//	N.n(0,0) = 0;
//	for(int s = 1 ; s != 487 ; ++s){
//		N.n(0,s) = Train(0,s);
//	}
//	N.update();
//
//	N.update();
//
//	Net check_grad = Net(vector<int>{487,500,125,1},fs);
//	const double target{10.};
//	const double v = pow(N.v(N.L().size()-1,0)-target,2);
//	double vbis{0};
//	const double eps{0.00000001};
//
//
//	for(int l = 0 ; l != N.L().size() -1 ; ++l ){
//		cout << l<<endl;
//		for(int s = 0 ; s != N.L()[l] ; ++s ){
//			for(int e = 0 ; e != N.L()[l+1]; ++e){
//				N.c(l,s,e) += eps; /* node 0 of layer 0 */
//				N.update();
//				vbis = pow(N.v(N.L().size()-1,0)-target,2);
//				N.c(l,s,e) -= eps; /* node 0 of layer 0 */
//				N.update();
//				check_grad.c(l,s,e) = (vbis-pow(N.v(N.L().size()-1,0)-target,2))/eps;
//			}
//		}
//	}
////
//	grad(N,G,target);
////
////	G.print();
//	cout << G.get_coeffs()-check_grad.get_coeffs();



	std::function<arma::mat(const arma::mat &, mt19937 & )> g_st = [&N,&G,&res_grad](const arma::mat & v, mt19937 & g){
	uniform_int_distribution<int> Dist(0,::end_train-10);
	const int batchSize{1};

	N.get_coeffs() = v;
	res_grad.fill(0.);
	int d{0};
	for(int i = 0 ; i!= batchSize ; ++i){
		d = Dist(g);
//		For each available date, we calculate a gradient and then we average
		N.n(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.n(0,s) = ::Train(d,s);
		}
		N.update();
		grad(N,G,::Train(d+10,0));
		res_grad += G.get_coeffs();
	}
//	cout << N.v(3,0)<<endl;
	return res_grad/(batchSize);
	};




//	dataframe dv0(N.get_coeffs().n_rows,1,"v5assets.csv",false);
//	vec v0 = dv0.getData();
	vec v0{N.get_coeffs().n_rows,fill::randn};

	auto start = std::chrono::high_resolution_clock::now();
	vec vinf = stochastic_descent(v0,g_st,0.000000000000001,0.01);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	dataframe initVec{vinf};
	initVec.write_csv("v1.csv");

	cout << norm( v0 - vinf)/v0.n_rows <<endl;

	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.n(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.n(0,s) = Train(d,s);
		}
		N.update();
		cout << N.v(3,0) << ' '<< Train(10+d,0)<<' ';
	}
	cout << "time" << elapsed.count();


	return 0;
}


