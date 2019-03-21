/*
 * Net.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: oliv
 */

#include "Net.h"
using namespace std;
using namespace arma;
Net::Net() : Net(vector<int>(0)) {

}

Net::Net(const std::vector<int>& layers):layers(layers) {
	int n_coeffs{0};
	int n_nodes{0};
	int temp{1};

	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
		if(it == layers.begin()){
			temp = *it;
			n_nodes += temp;
		}
		else{
			n_coeffs += temp * *it;
			temp = *it;
			n_nodes += temp;
		}
	}
	coeffs = vec(n_coeffs,fill::randu);
	nodevals = vec(n_nodes,fill::zeros);
}

double& Net::c(const int& layer_num, const int& start, const int& end) {
	int cumsum{0};
	for(int i = 1; i !=layer_num+1;++i){
		cumsum += layers[i]*layers[i-1];
	}
	return coeffs(cumsum+start*layers[layer_num+1]+end);
}

double Net::c(const int& layer_num, const int& start, const int& end) const{
	int cumsum{0};
	for(int i = 1; i !=layer_num+1;++i){
		cumsum += layers[i]*layers[i-1];
	}
	return coeffs(cumsum+start*layers[layer_num+1]+end);
}

double& Net::v(const int& layer_num, const int& i) {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodevals(cumsum+i);
}

double Net::v(const int& layer_num, const int& i) const {
	int cumsum{0};
	for(int i = 0; i !=layer_num;++i){
		cumsum += layers[i];
	}
	return nodevals(cumsum+i);
}

arma::vec& Net::get_coeffs() {
	return coeffs;
}

void Net::print() {
	int temp{0};
	cout << "VALUES"<<endl;
	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
			cout << nodevals.rows(temp,temp+*it-1)<<endl;
			temp += *it;
	}
	temp = 0;
	cout << "COEFFICIENTS"<<endl;
	for( vector<int>::const_iterator it = layers.begin(); it != layers.end() ; ++it){
		if(it == layers.begin()){
		}
		else{
			cout << coeffs.rows(temp,temp+ *it * *(it-1) -1)<<endl;
			temp += *it * *(it-1);
		}
	}
}

void Net::update() {
	for(int l = 1 ; l != layers.size() ; l++){
		for(int end = 0 ; end != layers[l] ; ++end){
			v(l,end) = 0;
			for(int start = 0 ; start != layers[l-1] ; ++start){
				v(l,end) += c(l-1,start,end)*v(l-1,start);
//				cout << v(l,end) << endl;
			}
//			v(l,end) = v(l,end)>0 ? v(l,end):0; /* max(v,0) */
		}
	}
}

Net::~Net() {
	// TODO Auto-generated destructor stub
}

