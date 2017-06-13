cdef class Classifier(_JubatusBase):
    cdef _Classifier *_handle
    cdef object classes_

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Classifier(config)
        self.classes_ = None
        self._type, self._model_ver = b'classifier', 1

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
        cdef string l
        cdef datum d
        if isinstance(data, list):
            if len(data) == 0:
                return 0
            if isinstance(data[0], LabeledDatum):
                for r in data:
                    label, datum = r.label, r.data
                    l = label.encode('utf8')
                    datum_py2native(datum, d)
                    self._handle.train(l, d)
            elif isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
                for label, datum in data:
                    l = label.encode('utf8')
                    datum_py2native(datum, d)
                    self._handle.train(l, d)
            else:
                raise ValueError
            return len(data)
        else:
            raise ValueError

    def classify(self, data):
        cdef datum d
        cdef vector[classify_result_elem] r
        if isinstance(data, list):
            ret = []
            for x in data:
                datum_py2native(x, d)
                r = self._handle.classify(d)
                ret.append([
                    EstimateResult(r[i].label.decode('utf8'), r[i].score)
                    for i in range(r.size())
                ])
            return ret
        else:
            raise ValueError

    def get_labels(self):
        cdef vector[pair[string, uint64_t]] ret = self._handle.get_labels()
        return {ret[i].first.decode('utf8'): ret[i].second for i in range(ret.size())}

    def set_label(self, new_label):
        return self._handle.set_label(new_label.encode('utf8'))

    def delete_label(self, target_label):
        return self._handle.delete_label(target_label.encode('utf8'))

    def fit(self, X, y):
        self._handle.clear()
        self.classes_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        import numpy as np
        if len(X.shape) != 2 or len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError('invalid shape')
        cdef int i, j, max_label
        cdef rows = X.shape[0]
        cdef datum d
        cdef vector[string] cache
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        if not (is_ndarray or is_csr):
            raise ValueError
        if self.classes_ is None:
            self.classes_ = []
        max_label = len(self.classes_) - 1
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(X, i, d, cache)
            else:
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
            for j in range(cache.size(), y[i] + 1):
                cache.push_back(lexical_cast[string, int](j))
            if max_label < y[i]:
                max_label = y[i]
            self._handle.train(cache[y[i]], d)
        for j in range(len(self.classes_), max_label + 1):
            self.classes_.append(j)
        return self

    def decision_function(self, X):
        import numpy as np
        if len(X.shape) != 2:
            raise ValueError('invalid X.shape')
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        if not (is_ndarray or is_csr):
            raise ValueError
        cdef int i, j, k
        cdef double score
        cdef datum d
        cdef vector[string] cache
        cdef vector[classify_result_elem] r
        cdef int rows = X.shape[0]
        ret = None
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(X, i, d, cache)
            else:
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
            r = self._handle.classify(d)
            if r.size() == 0:
                return np.zeros(rows)
            if r.size() == 2:
                if ret is None:
                    ret = np.zeros(rows)
                if r[0].score < r[1].score:
                    j = lexical_cast[int, string](r[1].label)
                    score = r[1].score
                else:
                    j = lexical_cast[int, string](r[0].label)
                    score = r[0].score
                if j == 0:
                    ret[i] = -score
                else:
                    ret[i] = score
            else:
                if ret is None:
                    ret = np.zeros((rows, r.size()))
                for j in range(r.size()):
                    k = lexical_cast[int, string](r[j].label)
                    ret[i,k] = r[j].score
        return ret

    def predict(self, X):
        import numpy as np
        if len(X.shape) != 2:
            raise ValueError('invalid X.shape')
        cdef int is_ndarray = isinstance(X, np.ndarray)
        cdef int is_csr = (type(X).__name__ == 'csr_matrix')
        if not (is_ndarray or is_csr):
            raise ValueError
        cdef int i, j, max_j
        cdef double max_score
        cdef datum d
        cdef vector[string] cache
        cdef vector[classify_result_elem] r
        cdef int rows = X.shape[0]
        IF NUMPY:
            cdef c_np.ndarray[c_np.int32_t, ndim=1] ret = np.zeros(rows, dtype=np.int32)
        ELSE:
            ret = np.zeros(rows, dtype=np.int32)
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(X, i, d, cache)
            else:
                csr_to_datum(X.data, X.indices, X.indptr, i, d, cache)
            r = self._handle.classify(d)
            if r.size() == 0:
                break
            max_j, max_score = 0, r[0].score
            for j in range(1, r.size()):
                if r[j].score > max_score:
                    max_j, max_score = j, r[j].score
            ret[i] = lexical_cast[int, string](r[max_j].label)
        return ret
