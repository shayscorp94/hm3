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


int main(){
//	Defines fully connected neural net : layer0 : 2 nodes / layer 1 : 1 node
	Net N = Net(vector<int>{8,4,2,1});
//	Set nodes value for input layer
	for(int s = 0 ; s != 8 ; s++){
		N.v(0,s) = rand()%10; /* node 0 of layer 0 */
	}

	N.print();

	N.update();
	cout << "\n\nAfter update\n\n";
	N.print();

	Net G = grad(N,35.519);
	G.print();
	return 0;
}




