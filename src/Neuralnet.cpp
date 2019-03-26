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
double Drelu(const double & d){
	return d > 0 ? 1 : 0;
}

Net grad(const Net & N,const double & target){
	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	Net G{layers};
//	Deal with end node if net is not empty
	if(n_layers > 0){
//		G.v(n_layers-1,0) = 2*Drelu( N.v(n_layers-1,0) )*( relu(  N.v(n_layers-1,0)  ) -target);
//		In fact the Drelu is useless because 1 when relu > 0 , 0 when relu = 0
		G.v(n_layers-1,0) = 2*( relu(  N.v(n_layers-1,0)  ) -target);

	}

	for(int l = layers.size()-2 ; l != -1 ; --l){
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
				G.v(l,start) += N.c(l,start,end)*G.v(l+1,end);
			}
		}
	}
	return G;
}

arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & etha,const double & eps){
	mat v{v0};
	mat g = grad(v);
//	double old_n = norm(g);
	double eth = etha;
//	double temp = old_n;
	const int maxIt{1000};
	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{
			v = v-eth*g;
			g = grad(v);
//			temp = norm(g);
//			if(temp > old_n){
//				eth /= 10.;
//				cout << 'down_step'<<endl;
//			}
//			old_n = temp;
//			cout << "grad" << norm(g) << endl;
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}

//arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
//	mat v{v0};
//	mat g = grad(v);
//	const int maxIt{1000};
//	for(int i = 0 ; i != maxIt ; ++i ){
//		if(norm(g) < eps){
//			cout << "numit" << i <<' ' << norm(g) <<'\n';
//			return v;
//		}
//		else{
//			v = v-eth*g;
//			g = grad(v);
////			cout << norm(g) << endl;
//		}
//	}
//	cout << "reached max it" << norm(g) << '\n';
//	return v;
//}


arma::mat acc_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
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

	const int nassets{10};
	const int nlines{756};
	const double end_train{10};
	dataframe Data{756,nassets,"cleanIndex.csv"};
	mat Train = Data.getData().rows(0,end_train);

//	Define fully connected neural net : layer0 : 500 nodes / layer 1 : 250  nodes / layer 2 : 125 nodes / layer 3 : 1 node

//	For layer 0 we will add 0s because we only have 486 assets.

	Net N = Net(vector<int>{500,250,125,1});
	Net G = Net(vector<int>{500,250,125,1}); /* we do not care of the target we just want the structure*/
	vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);

	std::function<arma::mat(const arma::mat & )> g = [&Train,&N,&G,&res_grad,&end_train](const arma::mat & v){
	N.get_coeffs() = v;
	res_grad.fill(0.);
	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.v(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.v(0,s) = Train(d,s);
		}
		N.update();
		G = grad(N,Train(d+10,0));
		res_grad += G.get_coeffs();
	}
//	cout << N.v(3,0)<<endl;
	return res_grad/(end_train+1-10);
	};
	Net T = Net(vector<int>{500,250,125,1}); /* we do not care of the target we just want the structure*/

	vec v0 = vec(T.get_coeffs().n_rows,fill::randn);
	auto start = std::chrono::high_resolution_clock::now();
	grad_descent(v0,g,0.0000001,0.01);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;

	cout << N.v(3,0)- Train(10,0)<<' '<< elapsed.count();
	return 0;
}







