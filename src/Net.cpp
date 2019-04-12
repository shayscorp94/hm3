#include "Net.h"
#include "math.h"
#include <random>
using namespace std;
using namespace arma;


Net::Net(const std::vector<int> & layers, std::vector<double(*)(const double&)> fs)
	:layers(layers), fs(fs)
{
	int n_coeffs{ 0 };
	int n_nodes{ 0 };
	int temp{ 1 };

	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
		if (it == layers.begin()) {
			temp = *it;
			n_nodes += temp;
		}
		else {
			n_coeffs += temp ; //change 1
			temp = *it;
			n_nodes += temp;
		}
	}
	//	coeffs = vec(n_coeffs, fill::randn);
	////	cout << coeffs(0)<<endl;
	//	temp = 0;
	//	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
	//		if (it == layers.begin()) {
	//		}
	//		else {
	//			coeffs.rows(temp, temp + *it * *(it - 1) - 1) *= sqrt(2./ *(it - 1));
	////			cout << 1/ *(it - 1) << endl;
	//			temp += *it * *(it - 1);
	//		}
	//	}
	coeffs = vec(n_coeffs, fill::zeros);
	mt19937 g;
	std::normal_distribution<double> d;
	//	cout << coeffs(0)<<endl;
	temp = 0;
	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
		if (it == layers.begin()) {
		}
		else {
			for (int i = temp; i != temp + *(it - 1); ++i) { //change 2
				coeffs(i) = d(g)*sqrt(2. / *(it - 1));
			}
			//			cout << 1/ *(it - 1) << endl;
			temp += *(it - 1); //change 3
		}
	}
	//	cout << coeffs(0)<<endl;

	nodevals = vec(n_nodes, fill::zeros);
	nodes = vec(n_nodes, fill::zeros);

}

double& Net::c(const int& layer_num, const int& start, const int& end) {
	int cumsum{ 0 };
	for (int i = 1; i != layer_num + 1; ++i) {
		cumsum += layers[i - 1];  //change 4
	}
	return coeffs(cumsum + start); //change 5
}

double Net::c(const int& layer_num, const int& start, const int& end) const {
	int cumsum{ 0 };
	for (int i = 1; i != layer_num + 1; ++i) {
		cumsum += layers[i - 1]; //change 6
	}
	return coeffs(cumsum + start); //change 7
}

double& Net::v(const int& layer_num, const int& i) {
	int cumsum{ 0 };
	for (int i = 0; i != layer_num; ++i) {
		cumsum += layers[i];
	}
	return nodevals(cumsum + i);
}

double Net::v(const int& layer_num, const int& i) const {
	int cumsum{ 0 };
	for (int i = 0; i != layer_num; ++i) {
		cumsum += layers[i];
	}
	return nodevals(cumsum + i);
}

double& Net::n(const int& layer_num, const int& i) {
	int cumsum{ 0 };
	for (int i = 0; i != layer_num; ++i) {
		cumsum += layers[i];
	}
	return nodes(cumsum + i);
}

double Net::n(const int& layer_num, const int& i) const {
	int cumsum{ 0 };
	for (int i = 0; i != layer_num; ++i) {
		cumsum += layers[i];
	}
	return nodes(cumsum + i);
}

arma::vec& Net::get_coeffs() {
	return coeffs;
}

void Net::print() {
	int temp{ 0 };
	cout << "VALUES" << endl;
	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
		cout << nodevals.rows(temp, temp + *it - 1) << endl;
		temp += *it;
	}
	temp = 0;
	cout << "COEFFICIENTS" << endl;
	for (vector<int>::const_iterator it = layers.begin(); it != layers.end(); ++it) {
		if (it == layers.begin()) {
		}
		else {
			cout << coeffs.rows(temp, temp + *(it - 1) - 1) << endl; //change 8
			temp += *(it - 1); //change 9
		}
	}
}

void Net::update() {
	if (layers.size() > 0) {
		for (int end = 0; end != layers[0]; ++end) {
			v(0, end) = fs[0](n(0, end)); /* function of layer */
	//		cout << n(0,end)<<endl;
		}
	}
	for (int l = 1; l != layers.size() - 1; l++) { //change 9 - remember to change for the next column
		int start = 0;
		for (int end = 0; end != layers[l]; ++end) {
			n(l, end) = 0;
			n(l, end) = c(l - 1, start, end)*v(l - 1, start) + c(l - 1, start + 1, end)*v(l - 1, start + 1); //change 10
			v(l, end) = fs[l](n(l, end)); /* function of layer */ //change 11
			start += 2; // change 12
		}
	}
	// Now l = layers.size() - 1  //change 13
	int l = layers.size() - 1;
	for (int end = 0; end != layers[l]; ++end) {
		n(l, end) = 0;
		for (int start = 0; start != layers[l - 1]; ++start) {
			n(l, end) += c(l - 1, start, end)*v(l - 1, start);
		}
		v(l, end) = fs[l](n(l, end)); /* function of layer */
	}
}

Net::~Net() {
	// TODO Auto-generated destructor stub
}