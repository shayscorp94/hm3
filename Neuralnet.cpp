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
//	Defines fully connected neural net : layer0 : 2 nodes / layer 1 : 1 node
	Net N = Net(vector<int>{4,2,1});
//	Set nodes value for input layer
	N.v(0,0) = 2; /* node 0 of layer 0 */
	N.v(0,1) = 4;/* node 1 of layer 0 */
	N.v(0,2) = 3;/* node 2 of layer 0 */
	N.v(0,3) = 1;/* node 3 of layer 0 */

//  Set coefficients values between layers
//	by default set random normal
	N.c(0,2,1) = 1; /* coeff from layer0,node2 to the node1 of the next layer (layer1)*/

	N.print();

	N.update();
	cout << "\n\nAfter update\n\n";
	N.print();
	return 0;
}

