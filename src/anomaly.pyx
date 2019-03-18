cdef class Anomaly(_JubatusBase):
    cdef _Anomaly *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Anomaly(config)
        self._type, self._model_ver = b'anomaly', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def clear_row(self, id_):
        self._handle.clear_row(id_.encode('utf8'))
        return True

    def add(self, row):
        cdef datum d
        cdef pair[string, double] r
        datum_py2native(row, d)
        r = self._handle.add(d)
        return AnomalyIdWithScore(r.first.decode('utf8'), r.second)

    def add_bulk(self, data):
        cdef datum d
        cdef vector[pair[string, datum]] vec
        cdef vector[string] ret
        for x in data:
            datum_py2native(x, d)
            vec.push_back(pair[string, datum](self._handle.get_next_id(), d))
        ret = self._handle.add_bulk(vec)
        return [i.decode('ascii') for i in ret]

    def update(self, id_, row):
        cdef datum d
        datum_py2native(row, d)
        return self._handle.update(id_.encode('utf8'), d)

    def overwrite(self, id_, row):
        cdef datum d
        datum_py2native(row, d)
        return self._handle.overwrite(id_.encode('utf8'), d)

    def calc_score(self, row):
        cdef datum d
        datum_py2native(row, d)
        return self._handle.calc_score(d)

    def get_all_rows(self):
        cdef vector[string] ret = self._handle.get_all_rows()
        return [i.decode('ascii') for i in ret]

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }

    def fit(self, X):
        self.partial_fit(X)

    def partial_fit(self, X):
        import numpy as np
        cdef datum d
        cdef vector[string] cache
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        cdef rows
        cdef vector[pair[string, datum]] vec
        if not (is_ndarray or is_csr):
            raise ValueError
        rows = X.shape[0]
        if is_ndarray:
            for i in range(rows):
                ndarray_to_datum(X, i, d, cache)
                vec.push_back(pair[string, datum](self._handle.get_next_id(), d))
        else:
            for i in range(rows):
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
                vec.push_back(pair[string, datum](self._handle.get_next_id(), d))
        self._handle.add_bulk(vec)

    def decision_function(self, X):
        import numpy as np
        cdef datum d
        cdef vector[string] cache
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        cdef rows
        if not (is_ndarray or is_csr):
            raise ValueError
        rows = X.shape[0]
        y = np.zeros((rows,))
        if is_ndarray:
            for i in range(rows):
                ndarray_to_datum(X, i, d, cache)
                y[i] = self._handle.calc_score(d)
        else:
            for i in range(rows):
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
                y[i] = self._handle.calc_score(d)
        return y
