from libcpp cimport bool

cdef extern from 'cudarray/reduction.hpp' namespace 'cudarray':
    enum ReduceOp:
        MAX_OP
        MEAN_OP
        MIN_OP
        SUM_OP

    enum ReduceToIntOp:
        ARGMAX_OP
        ARGMIN_OP

    void reduce[T](ReduceOp op, const T *a, unsigned int n, T *b)
    void reduce_mat[T](ReduceOp op, const T *a, unsigned int m, unsigned int n,
                       bool reduce_leading, T *b)

    void reduce_to_int[T](ReduceToIntOp op, const T *a, unsigned int n, int *b)
    void reduce_mat_to_int[T](ReduceToIntOp op, const T *a, unsigned int m,
                              unsigned int n, bool reduce_leading, int *b)
