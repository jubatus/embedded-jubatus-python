#include <vector>
#include <jubatus/util/concurrent/rwmutex.h>
#include "_helper.h"

using jubatus::util::lang::lexical_cast;

static std::vector<std::string> cache;
static jubatus::util::concurrent::rw_mutex mutex;

void allocate_number_string(std::size_t max_num) {
    jubatus::util::concurrent::scoped_wlock lk(mutex);
    for (std::size_t i = cache.size(); i <= max_num; ++i)
        cache.push_back(lexical_cast<std::string>(i));
}

const std::string& get_number_string(std::size_t num) {
    {
        jubatus::util::concurrent::scoped_rlock lk(mutex);
        if (cache.size() > num)
            return cache[num];
    }
    {
        jubatus::util::concurrent::scoped_wlock lk(mutex);
        for (std::size_t i = cache.size(); i <= num; ++i)
            cache.push_back(lexical_cast<std::string>(i));
        return cache[num];
    }
}

const std::string& get_number_string_fast(std::size_t num) {
    jubatus::util::concurrent::scoped_rlock lk(mutex);
    return cache[num];
}

void ndarray_to_datum(const PyObject *obj, size_t i, datum& out) {
    const PyArrayObject *ary = (const PyArrayObject*)obj;
    const char *p = PyArray_BYTES(ary) + (i * PyArray_STRIDE(ary, 0));
    npy_intp stride = PyArray_STRIDE(ary, 1);

    out.string_values_.clear();
    out.num_values_.clear();
    out.binary_values_.clear();

    jubatus::util::concurrent::scoped_rlock lk(mutex);
    for (npy_intp j = 0; j < PyArray_DIM(ary, 1); j ++) {
        double v = *(double*)(p + j * stride);
        if (v != 0.0)
            out.num_values_.push_back(std::make_pair(cache[j], v));
    }
}

void csr_to_datum(const PyObject *obj0, const PyObject *obj1,
                  const PyObject *obj2, size_t i, datum& out) {
    const PyArrayObject *data = (const PyArrayObject*)obj0;
    const PyArrayObject *indices = (const PyArrayObject*)obj1;
    const PyArrayObject *indptr = (const PyArrayObject*)obj2;

    out.string_values_.clear();
    out.num_values_.clear();
    out.binary_values_.clear();

    int j = *(int32_t*)PyArray_GETPTR1(indptr, i);
    int k = *(int32_t*)PyArray_GETPTR1(indptr, i + 1);

    jubatus::util::concurrent::scoped_rlock lk(mutex);
    for (npy_intp l = j; l < k; l ++) {
        int32_t x = *(int32_t*)PyArray_GETPTR1(indices, l);
        double v = *(double*)PyArray_GETPTR1(data, l);
        out.num_values_.push_back(std::make_pair(cache[x], v));
    }
}
