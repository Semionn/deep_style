#ifndef ARRAY_OPS_HPP_
#define ARRAY_OPS_HPP_

namespace cudarray {

template<typename T>
void transpose(const T *a, unsigned int n, unsigned int m, T *b);

template<typename Ta, typename Tb>
void as(const Ta *a, unsigned int n, Tb *b);

template<typename T>
void fill(T *a, unsigned int n, T alpha);

template<typename T>
void copy(const T *a, unsigned int n, T *b);

template<typename T>
void to_device(const T *a, unsigned int n, T *b);

template<typename T>
void to_host(const T *a, unsigned int n, T *b);

}

#endif  // ARRAY_OPS_HPP_
