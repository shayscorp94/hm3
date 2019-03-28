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

double I(const double & t){
	return t;
}

double f(const double & t){
	return t*t;
}


int main(){
	vector<int> layers{2,1};
	vector< double (*)(const double &)> v{I,f};

	Net N(layers,v,v);
	N.n(0,1) = 0;
	N.n(0,2) = 1;
	N.update();
	N.print();

	return 0;
}


