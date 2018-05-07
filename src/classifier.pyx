cdef class Classifier(_JubatusBase):
    cdef _Classifier *_handle
    cdef object _classes

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Classifier(config)
        self._classes = None
        self._type, self._model_ver = b'classifier', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        cdef vector[pair[string, uint64_t]] labels
        ret = self._handle.load(x, self._type, self._model_ver)
        labels = self._handle.get_labels()
        tmp = set(range(len(labels)))
        for pair in labels:
            try:
                tmp.remove(int(pair.first.decode('ascii')))
            except Exception:
                break
        if len(tmp) == 0:
            self._classes = list(range(len(labels)))
        return ret

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

    @property
    def classes_(self):
        return self._classes

    def fit(self, X, y):
        self._handle.clear()
        self._classes = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        import numpy as np
        cdef unsigned int max_label, i, rows = X.shape[0]
        cdef datum d
        cdef int is_ndarray = check_ndarray_csr_type(X)
        if len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError('invalid shape')
        if y.dtype.kind not in ('i', 'u'):
            raise ValueError('invalid y.dtype')
        if self._classes is None:
            self._classes = []
        max_label = max(len(self._classes) - 1, max(y))
        allocate_number_string(max(X.shape[1], max_label))
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(<const PyObject*>X, i, d)
            else:
                csr_to_datum(<const PyObject*>X.data,
                             <const PyObject*>X.indices,
                             <const PyObject*>X.indptr, i, d)
            self._handle.train(get_number_string_fast(y[i]), d)
        for i in range(len(self._classes), max_label + 1):
            self._classes.append(i)
        return self

    def decision_function(self, X):
        import numpy as np
        cdef int is_ndarray = check_ndarray_csr_type(X)
        cdef int k
        cdef size_t j
        cdef double score
        cdef datum d
        cdef vector[classify_result_elem] r
        cdef unsigned int i, rows = X.shape[0]
        ret = None
        allocate_number_string(X.shape[1])
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(<const PyObject*>X, i, d)
            else:
                csr_to_datum(<const PyObject*>X.data,
                             <const PyObject*>X.indices,
                             <const PyObject*>X.indptr, i, d)
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
        cdef int is_ndarray = check_ndarray_csr_type(X)
        cdef size_t j, max_j
        cdef double max_score
        cdef datum d
        cdef vector[classify_result_elem] r
        cdef unsigned int i, rows = X.shape[0]
        cdef c_np.ndarray[c_np.int32_t, ndim=1] ret = np.zeros(rows, dtype=np.int32)
        allocate_number_string(X.shape[1])
        for i in range(rows):
            if is_ndarray:
                ndarray_to_datum(<const PyObject*>X, i, d)
            else:
                csr_to_datum(<const PyObject*>X.data,
                             <const PyObject*>X.indices,
                             <const PyObject*>X.indptr, i, d)
            r = self._handle.classify(d)
            if r.size() == 0:
                break
            max_j, max_score = 0, r[0].score
            for j in range(1, r.size()):
                if r[j].score > max_score:
                    max_j, max_score = j, r[j].score
            ret[i] = lexical_cast[int, string](r[max_j].label)
        return ret
