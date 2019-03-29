/*
 * net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */


#include <iostream>
#include <armadillo>
#include <algorithm>    // std::random_shuffle

#include "Net.h"
#include "dataframe.h"

using namespace std;
using namespace arma;
using namespace vSpace;

double relu(const double & d){
	return d > 0 ? d : 0;
}

double Lrelu(const double & d){
	return d > 0 ? d : 0.01*d;
}

double DLrelu(const double & d){
	return d > 0 ? 1 : 0.01;
}



double I(const double & t){
	return t;
}

double One(const double & t){
	return 1;
}

double Drelu(const double & d){
	return d > 0 ? 1 : 0;
}


inline void grad(const Net & N,Net & G,const double & target){

	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	const vector<double (*)(const double &)> deriv=G.getFs();
//	Deal with end node if net is not empty
	if(n_layers > 0){
		G.v(n_layers-1,0) = 2*deriv[n_layers-1](N.n(n_layers-1,0))*( N.v(n_layers-1,0)  -target);
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
				G.v(l,start) += N.c(l,start,end)*deriv[l](N.v(l,start))*G.v(l+1,end);
			}
		}
	}
}

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

int main(){
//	Data processing

	const int nassets{487};
	const int nlines{756};
	const double end_train{11};
	dataframe Data{756,nassets,"cleanIndex.csv"};
	mat Train = Data.getData().rows(0,end_train);

//	Define fully connected neural net : layer0 : 500 nodes / layer 1 : 250  nodes / layer 2 : 125 nodes / layer 3 : 1 node

//	For layer 0 we will add 0s because we only have 486 assets.

	vector<double (*)(const double &)> fs{I,Lrelu,Lrelu,Lrelu};
	vector<double (*)(const double &)> ds{One,DLrelu,DLrelu,DLrelu};


	Net N = Net(vector<int>{487,250,125,1},fs);
	Net G = Net(vector<int>{487,250,125,1},ds); /* we do not care of the target we just want the structure*/
	vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);

	std::function<arma::mat(const arma::mat & )> g = [&Train,&N,&G,&res_grad,&end_train](const arma::mat & v){
	N.get_coeffs() = v;
	res_grad.fill(0.);
	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.n(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.n(0,s) = Train(d,s);
		}
		N.update();
//		cout <<N.v(3,0)<<endl;
		grad(N,G,Train(d+10,0));
		res_grad += G.get_coeffs();
	}
//	cout << N.v(3,0)<<endl;
	return res_grad/(end_train+1-10);
	};
	dataframe dv0(N.get_coeffs().n_rows,1,"v0.csv",false);
	vec v0 = dv0.getData();
//	vec v0(N.get_coeffs().n_rows,fill::randn);

	auto start = std::chrono::high_resolution_clock::now();
	vec vinf = grad_descent(v0,g,0.00000000001,0.01);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	dataframe initVec{vinf};
	initVec.write_csv("v1.csv");

	cout << norm( v0 - vinf)/v0.n_rows <<endl;

	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.v(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.v(0,s) = Train(d,s);
		}
		N.update();
		cout << N.v(3,0) << ' '<< Train(10+d,0)<<' ';
	}
	cout << "time" << elapsed.count();
	return 0;
}


