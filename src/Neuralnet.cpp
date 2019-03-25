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

arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
	mat v{v0};
	mat g = grad(v);
	const int maxIt{100};
	for(int i = 0 ; i != maxIt ; ++i ){
		if(norm(g) < eps){
			cout << "numit" << i <<' ' << norm(g) <<'\n';
			return v;
		}
		else{
			v = v-eth*g;
			g = grad(v);
		}
	}
	cout << "reached max it" << norm(g) << '\n';
	return v;
}



int main(){
//	Data processing

	const int nassets{487};
	const int nlines{756};
	const int end_train{10};
	dataframe Data{756,nassets,"cleanIndex.csv"};
	mat Train = Data.getData().rows(0,end_train);


//	Define fully connected neural net : layer0 : 500 nodes / layer 1 : 250  nodes / layer 2 : 125 nodes / layer 3 : 1 node

//	For layer 0 we will add 0 because we only have 486 assets.


//	Net N = Net(vector<int>{500,250,125,1});
//	Net G = grad(N,0); /* we do not care of the target we just want the structure*/


	std::function<arma::vec(const arma::vec & )> g = [Train](const arma::vec & v){
	Net N = Net(vector<int>{500,250,125,1});
	Net G = Net(vector<int>{500,250,125,1}); /* we do not care of the target we just want the structure*/
	vec res_grad = vec(G.get_coeffs().n_rows,fill::zeros);
	N.get_coeffs() = v;
	for(int d = 0 ; d != end_train+1-10 ; ++d){
//		For each available date, we calculate a gradient and then we average
		N.v(0,0) = 0;
		for(int s = 1 /* do not use the current index price*/ ; s != nassets ; ++s){
			N.v(0,s) = Train(d,s);
//			cout << s<<endl;

		}
		for(int s = nassets; s != 500 ; ++s){
			N.v(0,s) = 0;
//			cout << N.v(0,s)<<endl;
		}

		N.update();
		G = grad(N,Train(0,d+10));
		res_grad += G.get_coeffs();

	}
//	cout <<  res_grad<<endl;
	return res_grad;
	};
	cout <<1 <<endl;

	vec v0 = vec(500*250+250*125+125,fill::zeros);
	cout << g(v0);
	cout<<1<<endl;

//	cout << gradi.n_rows;
//	cout << grad_descent(N.get_coeffs(),g,0.0001,0.01)<<endl;
//	cout << N.v(N.L().size()-1,0);

	return 0;
}


//	Net check_grad = Net(vector<int>{8,4,2,1});
//	const double target{0.};
//	const double v = pow(N.v(4-1,0)-target,2);
//	double vbis{0};
//	const double eps{0.00000001};
//
//	for(int l = 0 ; l != N.L().size() -1 ; ++l ){
//		for(int s = 0 ; s != N.L()[l] ; ++s ){
//			for(int e = 0 ; e != N.L()[l+1]; ++e){
//				N.c(l,s,e) += eps; /* node 0 of layer 0 */
//				N.update();
//				vbis = pow(N.v(N.L().size()-1,0)-target,2);
//				N.c(l,s,e) -= eps; /* node 0 of layer 0 */
//				N.update();
//				check_grad.c(l,s,e) = (vbis-v)/eps;
//			}
//		}
//	}
//
//	Net G = grad(N,target);
//
//	cout << G.get_coeffs() -check_grad.get_coeffs();


//	N.update();
//	cout <<"before descent" <<N.v(3,0)<<endl;
//	for(int i = 0 ; i != 100 ; ++i){
//	Net G = grad(N,0);
//	N.get_coeffs() -= 0.0001*G.get_coeffs();
//	N.update();
//	cout <<N.v(3,0)<<endl;
//
//	}
//	N.print();







