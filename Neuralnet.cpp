/*
 * net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */


#include <iostream>
#include <armadillo>

#include "Net.h"

using namespace std;
using namespace arma;

double relu(const double & d){
	return d > 0 ? d : 0;
}

Net grad(const Net & N,const double & target){
	const vector<int> & layers = N.L();
	const int n_layers = layers.size();
	Net G{layers};
//	Deal with end node if net is not empty
	if(n_layers > 0){
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
//
//arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
//	mat v{v0};
//	mat g = grad(v);
//	const int maxIt{100};
//	for(int i = 0 ; i != maxIt ; ++i ){
//		if(norm(g) < eps){
//			cout << "numit" << i <<' ' << norm(g) <<'\n';
//			return v;
//		}
//		else{
//			v = v-eth*g;
//			g = grad(v);
//		}
//	}
//	cout << "reached max it" << norm(g) << '\n';
//	return v;
//
//}


int main(){
//	Defines fully connected neural net : layer0 : 2 nodes / layer 1 : 1 node
	Net N = Net(vector<int>{8,4,2,1});
//	Set nodes value for input layer
	for(int s = 0 ; s != 8 ; s++){
		N.v(0,s) = s; /* node 0 of layer 0 */
	}


	N.update();
	cout <<"before descent" <<N.v(3,0)<<endl;
	for(int i = 0 ; i != 100 ; ++i){
	Net G = grad(N,0);
	N.get_coeffs() -= 0.0001*G.get_coeffs();
	N.update();
	cout <<N.v(3,0)<<endl;

	}
	N.print();





//	std::function<arma::mat(const arma::mat &)> g = [N](const arma::mat & v){};


	return 0;
}




