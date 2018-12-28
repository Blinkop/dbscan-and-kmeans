#ifndef _KMEANS_HPP_
#define _KMEANS_HPP_

#include <random>
#include <cstdint>
#include <cmath>

#include <iostream>
using namespace std;

namespace clustering
{
    template <typename T>
    struct Point
    {
    public:
        Point() { }
        Point(uint32_t n_dims) : _ndims(n_dims) { _data = new T[n_dims]; }
        ~Point() { if (_data) { delete[] _data; data = nullptr; }}

        T& operator[](int i) { return _data[i]; }
        Point<T>& operator+=(const Point<T>& rhs)
        {
            for (uint32_t i = 0; i < _ndims; i++)
                _data[i] += rhs._data[i];

            return *this;
        }
        Point<T>& operator/=(T rhs)
        {
            for (uint32_t i = 0; i < _ndims; i++)
                _data[i] = rhs._data[i] / rhs;

            return *this;
        }

        double euclidean_distance(const Point<T>& point)
        {
            T accum = 0.0;
            for (uint32_t i = 0; i < _ndims; i++)
                accum += (point._data[i] - _data[i]) * (point._data[i] - _data[i]);

            return std::sqrt(accum);
        }
    private:
        T* _data = nullptr;
        uint32_t _ndims = 0;
    };

    template <typename T>
    void kmenas(Point<T> data[],
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

        auto centroids = new Point<T>[num_clusters]; // cluster centroids
        auto indexes   = new uint32_t[num_clusters]; // indexes for initial centroids
        auto total_obs = new uint32_t[num_clusters]; // number of observations for each centroid
        auto relations = new uint32_t[num_observations]; // observation relation to centroid

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
            centroids[i] = Point<T>(num_features);
            for (uint32_t j = 0; j < num_features; j++)
                centroids[i][j] = data[indexes[i]][j];
        }

        uint32_t num_relations_changed = 1;

        while (true)
        {
            // get closest centroid for each observation
            for (uint32_t i = 0; i < num_observations; i++)
            {
                uint32_t index = 0;
                T min = ~0;
                for (uint32_t j = 0; j < num_clusters; j++)
                {
                    auto dist = data[i].euclidean_distance(centroids[j]);
                    if (dist < min)
                    {
                        index = j;
                        min = dist;
                    }
                }

                if (relations[i] != index)
                {
                    ++num_relations_changed;
                    relations[i] = index;
                }
            }

            if (!num_relations_changed)
                break;

            // updating centroids position
            for (uint32_t i = 0; i < num_clusters; i++)
            {
                Point<T> accum(num_features);
                for (uint32_t j = 0; j < num_observations; j++)
                {
                    if (relations[j] == i)
                    {
                        total_obs[i]++;
                        accum += data[j];
                    }
                }

                accum /= total_obs[i];

                for (uint32_t j = 0; j < num_features; j++)
                    centroids[i][j] = accum[j];
            }
        }
    }
}

#endif // _KMEANS_HPP_
