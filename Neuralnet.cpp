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


int main(){
	Net N = Net(vector<int>{2,3,4});
	N.v(1,0)=2;
	N.c(0,0,0) = 4;
	N.print();
	return 0;
}

