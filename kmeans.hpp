#ifndef _KMEANS_HPP_
#define _KMEANS_HPP_

#include <random>
#include <cstdint>

#include <iostream>
using namespace std;

namespace clustering
{
    template <typename T>
    struct Point
    {
    public:
        Point() { }
        Point(uint32_t n_dims) { data = new T[n_dims]; }
        ~Point() { if (data) delete[] data; }

        T& operator[](int i) { return data[i]; }
    private:
        T* data = nullptr;
    };

    template <typename T>
    void kmenas(T* data[],
                uint32_t num_observations,
                uint32_t num_features,
                uint32_t num_clusters,
                uint32_t num_threads,
                uint32_t seed = 1337)
    {
        if (num_observations < num_clusters)
            return;
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint32_t> distr(0, num_observations - 1);

        auto centroids = new Point<T>[num_clusters];
        auto indexes   = new uint32_t[num_clusters];

        // generating unique random indexes for initial centroids
        for (int64_t i = 0; i < num_clusters; i++)
        {
            indexes[i] = distr(gen);
            for (int64_t j = 0; j < i; j++)
                if (indexes[i] == indexes[j])
                {
                    --i;
                    break;
                }
        }

        // initialize centroids with random points
        for (uint32_t i = 0; i < num_clusters; i++)
        {
            // TODO
        }
    }
}

#endif // _KMEANS_HPP_
