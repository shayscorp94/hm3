//#include <iostream>
//#include <armadillo>
//
//#include "Net.h"
//
//using namespace std;
//using namespace arma;
//
//double relu(const double & d) {
//	return d > 0 ? d : 0;
//}
//
//Net grad(const Net & N, const double & target) {
//	const vector<int> & layers = N.L();
//	const int n_layers = layers.size();
//	Net G{ layers };
//	//	Deal with end node if net is not empty
//	if (n_layers > 0) {
//		cout << "Last layer before grad is" << N.v(n_layers - 1, 0) << "and target is" << target <<endl;
//		G.v(n_layers - 1, 0) = 2 * (relu(N.v(n_layers - 1, 0)) - target);
//	}
//
//	for (int l = layers.size() - 2; l != -1; --l) {
//		//		Coeffs of G store partial diff with respect to that coeff
//		for (int start = 0; start != layers[l]; ++start) {
//			for (int end = 0; end != layers[l + 1]; ++end) {
//				G.c(l, start, end) = N.v(l, start)*G.v(l + 1, end);
//			}
//		}
//		//		Nodes of G store partial diff with respect to that node value
//		for (int start = 0; start != layers[l]; ++start) {
//			G.v(l, start) = 0;
//			for (int end = 0; end != layers[l + 1]; ++end) {
//				G.v(l, start) += N.c(l, start, end)*G.v(l + 1, end);
//			}
//		}
//	}
//	return G;
//}
////
////arma::mat grad_descent(const arma::mat& v0, std::function<arma::mat(const arma::mat &)> & grad,const double & eth,const double & eps){
////	mat v{v0};
////	mat g = grad(v);
////	const int maxIt{100};
////	for(int i = 0 ; i != maxIt ; ++i ){
////		if(norm(g) < eps){
////			cout << "numit" << i <<' ' << norm(g) <<'\n';
////			return v;
////		}
////		else{
////			v = v-eth*g;
////			g = grad(v);
////		}
////	}
////	cout << "reached max it" << norm(g) << '\n';
////	return v;
////
////}
//
//
//int main() {
//	Net N = Net(vector<int>{1, 4, 4, 1});
//	double x[100];
//	for (int i = 0; i < 500; i++)
//	{
//		x[i] = i*0.01;
//	}
//	//	Set nodes value for input layer
//	//	for (int s = 0; s != 1; s++) {
//	//	N.v(0, s) = 0.01; /* node 0 of layer 0 */
//	// }
//
//	N.v(0, 0) = x[0];
//	N.update();
//	cout << "before descent" << N.v(3, 0) << endl;
//	for (int i = 1; i != 500; ++i) {
//		N.v(0, 0) = x[i];
//		N.update();
//		Net G = grad(N, x[i]*2);
//		N.get_coeffs() -= 0.00001*G.get_coeffs();
//		N.update();
//		cout << N.v(3, 0) << endl;
//
//	}
//	N.print();
//
//
//
//
//
//	//	std::function<arma::mat(const arma::mat &)> g = [N](const arma::mat & v){};
//
//
//	return 0;
//}
