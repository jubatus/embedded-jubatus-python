cdef datum_py2native(pd, datum& d):
    d.string_values_.clear()
    for k, v in pd.string_values:
        k = k.encode('utf8')
        v = v.encode('utf8')
        d.string_values_.push_back(<tuple>(k, v))
    d.num_values_.clear()
    for k, v in pd.num_values:
        k = k.encode('utf8')
        d.num_values_.push_back(<tuple>(k, v))
    d.binary_values_.clear()
    for k, v in pd.binary_values:
        k = k.encode('utf8')
        d.binary_values_.push_back(<tuple>(k, v))

cdef datum_native2py(datum& d):
    ret = Datum()
    for i in range(d.string_values_.size()):
        k = d.string_values_[i].first.decode('utf8')
        v = d.string_values_[i].second.decode('utf8')
        ret.add_string(k, v)
    for i in range(d.num_values_.size()):
        k = d.num_values_[i].first.decode('utf8')
        v = d.num_values_[i].second
        ret.add_number(k, v)
    for i in range(d.binary_values_.size()):
        k = d.binary_values_[i].first.decode('utf8')
        v = d.binary_values_[i].second
        ret.add_binary(k, v)
    return ret

IF NUMPY:

    cdef ndarray_to_datum(c_np.ndarray[c_np.float64_t, ndim=2] X, int i, datum& d):
        d.string_values_.clear()
        d.num_values_.clear()
        d.binary_values_.clear()

        cdef int j
        for j in range(X.shape[1]):
            d.num_values_.push_back(<tuple>(lexical_cast[string, int](j), X[i, j]))

    cdef csr_to_datum(c_np.ndarray[c_np.float64_t, ndim=1] data,
                      c_np.ndarray[c_np.int32_t, ndim=1] indices,
                      c_np.ndarray[c_np.int32_t, ndim=1] indptr,
                      int i, datum& d):
        d.string_values_.clear()
        d.num_values_.clear()
        d.binary_values_.clear()

        cdef int j = indptr[i]
        cdef int k = indptr[i + 1]
        cdef int c, l
        for l in range(j, k):
            d.num_values_.push_back(<tuple>(
                lexical_cast[string, int](indices[l]),
                data[l]))

ELSE:

    cdef ndarray_to_datum(X, int i, datum& d):
        raise RuntimeError

    cdef csr_to_datum(data, indices, indptr, int i, datum& d):
        raise RuntimeError
