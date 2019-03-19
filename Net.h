/*
 * Net.h
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */

#ifndef NET_H_
#define NET_H_

#include <vector>
#include <armadillo>

class Net {
public:
	Net();
	Net(const std::vector<int> & layers); /* input a vector with the sizes of the layers, starting with the largest layer : l1 , l2 , lk */
	double & c(const int & layer, const int & start, const int & end); /* gives access to a coefficient, starting from the layer given as input, enables it to be modified */
	double & v(const int & layer, const int & i); /* gives access to a the node value from the layer given as input, enables it to be modified */
	arma::vec & get_coeffs(); /* returns reference to the whole vector of coefficients to be used for the gradient descent */
	void print();
	virtual ~Net();
private:
arma::vec coeffs; /* coefficients : vector of size  ( l1*l2 + l2*l3 + ... + l{k-1}*lk ) */
arma::vec nodevals; /*contains the values at each node, size is (l1+l2+...+lk) */
std::vector<int> layers;


};

#endif /* NET_H_ */
