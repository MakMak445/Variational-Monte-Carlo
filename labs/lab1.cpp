using namespace std;
#include<iostream>
#include<string>
#include<vector>
#include<cmath>
#include<fstream>
#include<complex>
#include<numbers>
#include<cmath>
#include<matplot/matplot.h>
vector<complex<double>> dft(const vector<double>& data, double max_time) {
    int N = data.size();
    double dt = N / max_time;
    vector<complex<double>> transformed;
    for (int k=0; k<N; k++) {
        complex<double> f_k = 0;
        for (int n=0; n<N; n++) {
            complex<double> z = std::polar(data[n], 2 * M_PI * k * n / N);
            f_k += z;
        }
        transformed.push_back(f_k);
    }
    return transformed;
}
int main() {
    using namespace matplot;
    return 0;

}