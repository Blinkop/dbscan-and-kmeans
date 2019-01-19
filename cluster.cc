#include "dbscan.hpp"
#include "kmeans.hpp"

#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace clustering;
using namespace dataframe;

namespace
{
    constexpr int test_data_size = 10000;
    constexpr int test_num_features = 4;
    constexpr int test_num_clusters = 6;
}

int main(int argc, char* argv[])
{
    // Point<float> data[test_data_size];

    // std::mt19937 gen(123);
    // std::normal_distribution<float> distr(0.0, 1.0);

    // for (auto i = 0; i < test_data_size; i++)
    // {
    //     auto tmp = Point<float>(test_num_features);
    //     for (auto j = 0; j < test_num_features; j++)
    //         tmp[j] = distr(gen);

    //     std::swap(data[i], tmp);
    // }

    // for (auto i = 0; i < test_data_size; i++)
    // {
    //     for (auto j = 0; j < test_num_features; j++)
    //         cout << data[i][j] << " ";

    //     cout << endl;
    // }
    auto max_threads = omp_get_max_threads();

    auto data = read_file<float>("X_train.txt");

    auto start = chrono::system_clock::now();
    kmenas(data.data, data.data_size, data.num_features, test_num_clusters, 1, 0, true);
    auto stop = chrono::system_clock::now();
    auto seq_time = (stop - start).count();
    cout << "Sequential time: " << seq_time << endl;

    start = chrono::system_clock::now();
    kmenas(data.data, data.data_size, data.num_features, test_num_clusters, max_threads, 0, true);
    stop = chrono::system_clock::now();
    auto par_time = (stop - start).count();
    cout << "Parallel time: " << par_time << endl << endl;

    cout << "Parallel boost coef: " << static_cast<double>(seq_time) / static_cast<double>(par_time) << endl;

    return 0;
}
