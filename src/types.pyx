cdef datum_py2native(pd, datum& d):
    d.string_values_.clear()
    for k, v in pd.string_values:
        k = k.encode('utf8')
        v = v.encode('utf8')
        d.string_values_.push_back(pair[string, string](k, v))
    d.num_values_.clear()
    for k, v in pd.num_values:
        k = k.encode('utf8')
        d.num_values_.push_back(pair[string, double](k, v))
    d.binary_values_.clear()
    for k, v in pd.binary_values:
        k = k.encode('utf8')
        d.binary_values_.push_back(pair[string, string](k, v))

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

cdef ndarray_to_datum(c_np.ndarray[c_np.float64_t, ndim=2] X, int i, datum& d, vector[string]& cache):
    d.string_values_.clear()
    d.num_values_.clear()
    d.binary_values_.clear()

    cdef int j
    for j in range(cache.size(), X.shape[1]):
        cache.push_back(lexical_cast[string, int](j))
    for j in range(X.shape[1]):
        if X[i, j] != 0.0:
            d.num_values_.push_back(pair[string, double](cache[j], X[i, j]))

cdef csr_to_datum(c_np.ndarray[c_np.float64_t, ndim=1] data,
                  c_np.ndarray[c_np.int32_t, ndim=1] indices,
                  c_np.ndarray[c_np.int32_t, ndim=1] indptr,
                  int i, datum& d, vector[string]& cache):
    d.string_values_.clear()
    d.num_values_.clear()
    d.binary_values_.clear()

    cdef int j = indptr[i]
    cdef int k = indptr[i + 1]
    cdef int l, m
    for l in range(j, k):
        for m in range(cache.size(), indices[l] + 1):
            cache.push_back(lexical_cast[string, int](m))
        d.num_values_.push_back(pair[string, double](cache[indices[l]], data[l]))

cdef props_py2native(p, prop_t& out):
    for k, v in p.items():
        out.insert(pair[string, string](k.encode('utf8'), v.encode('utf8')))

cdef props_native2py(prop_t& p):
    r = {}
    for it in p:
        r[it.first.decode('utf8')] = it.second.decode('utf8')
    return r

cdef edges_native2py(const vector[edge_id_t]& edges):
    ret = []
    for i in range(edges.size()):
        ret.append(edges[i])
    return ret

cdef preset_query_py2native(query, preset_query& q):
    for x in query.edge_query:
        q.edge_query.push_back(pair[string, string](
            x.from_id.encode('ascii'), x.to_id.encode('ascii')))
    for x in query.node_query:
        q.node_query.push_back(pair[string, string](
            x.from_id.encode('ascii'), x.to_id.encode('ascii')))
