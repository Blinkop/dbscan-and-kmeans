#ifndef _KMEANS_HPP_
#define _KMEANS_HPP_

#include <random>
#include <cstdint>
#include <limits>
#include <cmath>
#include <cstring>

#include <iostream>

#include <omp.h>

#include "dataframe.hpp"

using std::uint32_t;
using dataframe::Point;

namespace clustering
{
    template <typename T>
    void kmenas(Point<T> data[],
                uint32_t num_observations,
                uint32_t num_features,
                uint32_t num_clusters,
                uint32_t num_jobs,
                uint32_t max_iter = 0,
                bool verbose = false,
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

        for (uint32_t iteration = 0; iteration < max_iter || !max_iter; iteration++)
        {
            uint32_t num_relations_changed = 0;

            // get closest centroid for each observation
            #pragma omp parallel for default(none)\
                    shared(relations, data, centroids, num_observations, num_clusters)\
                    schedule(static)\
                    num_threads(num_jobs)\
                    reduction(+:num_relations_changed)
            for (uint32_t i = 0; i < num_observations; i++)
            {
                uint32_t index = 0;
                T min = std::numeric_limits<T>::max();
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

            if (verbose)
            {
                std::cout << "Iteration: " << iteration << std::endl;
                std::cout << "Relations changed:  " << num_relations_changed << std::endl;
            }

            if (!num_relations_changed)
                break;

            // updating centroids position
            #pragma omp parallel for default(none)\
                    shared(relations, data, centroids, total_obs, num_observations, num_clusters, num_features)\
                    schedule(dynamic)\
                    num_threads(num_jobs)
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
