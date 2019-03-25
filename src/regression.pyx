cdef class Regression(_JubatusBase):
    cdef _Regression *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Regression(config)
        self._type, self._model_ver = b'regression', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def train(self, data):
        cdef datum d
        if isinstance(data, list):
            if len(data) == 0:
                return 0
            if isinstance(data[0], ScoredDatum):
                for r in data:
                    score = r.score
                    datum_py2native(r.data, d)
                    self._handle.train(score, d)
            elif isinstance(data[0], (tuple, list)):
                for score, datum in data:
                    datum_py2native(datum, d)
                    self._handle.train(score, d)
            else:
                raise ValueError
            return len(data)
        else:
            raise ValueError

    def estimate(self, data):
        cdef datum d
        if isinstance(data, list):
            ret = []
            for datum in data:
                datum_py2native(datum, d)
                ret.append(self._handle.estimate(d))
            return ret
        else:
            raise ValueError

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }

    def fit(self, X, y):
        self.clear()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        import numpy as np
        if len(X.shape) != 2 or len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError('invalid shape')
        cdef int i
        cdef double score
        cdef rows = X.shape[0]
        cdef datum d
        cdef vector[string] cache
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        if not (is_ndarray or is_csr):
            raise ValueError
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(X, i, d, cache)
            else:
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
            score = y[i]
            self._handle.train(score, d)
        return self

    def predict(self, X):
        import numpy as np
        if len(X.shape) != 2:
            raise ValueError('invalid shape')
        cdef int i
        cdef rows = X.shape[0]
        cdef datum d
        cdef vector[string] cache
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        if not (is_ndarray or is_csr):
            raise ValueError
        ret = np.zeros([rows], dtype=np.float64)
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(X, i, d, cache)
            else:
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
            ret[i] = self._handle.estimate(d)
        return ret
