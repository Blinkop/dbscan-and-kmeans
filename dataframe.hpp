#ifndef _DATAFRAME_HPP_
#define _DATAFRAME_HPP_

#include <cstring>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <vector>

using std::uint32_t;

namespace dataframe
{
    template <typename T>
    struct Point
    {
    public:
        Point() { }
        Point(uint32_t n_dims) : _ndims(n_dims) { _data = new T[n_dims]; }
        ~Point() { if (_data) { delete[] _data; _data = nullptr; }}

        //rule of five
        Point<T>(const Point<T>& rhs)
        {
            _ndims = rhs._ndims;
            _data = new T[_ndims];

            std::memcpy(_data, rhs._data, _ndims * sizeof(T));
        }
        Point<T>(Point<T>&& rhs) noexcept
        {
            _data = rhs._data;
            _ndims = rhs._ndims;

            rhs._data = nullptr;
            rhs._ndims = 0;
        }
        Point<T>& operator=(const Point<T>& rhs)
        {
            _ndims = rhs._ndims;
            _data = new T[_ndims];

            std::memcpy(_data, rhs._data, _ndims * sizeof(T));

            return *this;
        }
        Point<T>& operator=(Point<T>&& rhs) noexcept
        {
            _data = rhs._data;
            _ndims = rhs._ndims;

            rhs._data = nullptr;
            rhs._ndims = 0;

            return *this;
        }

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
                _data[i] = _data[i] / rhs;

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
    struct DataFrame
    {
        Point<T>* data;
        uint32_t data_size;
        uint32_t num_features;
    };

    template <typename T>
    DataFrame<T> read_file(const std::string& filename)
    {
        DataFrame<T> result;

        std::ifstream file(filename);
        std::string buffer;

        uint32_t num_features = 0;
        uint32_t data_size = 0;

        if (std::getline(file, buffer))
        {
            std::istringstream iss(buffer);
            T tmp;

            while (iss >> tmp)
                num_features++;
        }

        std::vector<Point<T>> points;

        if (num_features)
        {
            do
            {
                auto point = Point<T>(num_features);
                std::istringstream iss(buffer);

                for (uint32_t i = 0; i < num_features; i++)
                    iss >> point[i];

                points.push_back(point);
                data_size++;
            } while (std::getline(file, buffer));
        }

        result.data = new Point<T>[data_size];
        result.data_size = data_size;
        result.num_features = num_features;

        for (uint32_t i = 0; i < data_size; i++)
            result.data[i] = points[i];

        return result;
    }
}

#endif // _DATAFRAME_HPP_
