cdef class Weight(_JubatusBase):
    cdef _Weight *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Weight(config)
        self._type, self._model_ver = b'weight', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def update(self, data):
        cdef datum d
        cdef sfv_t r
        datum_py2native(data, d)
        r = self._handle.update(d)
        return _to_features(r)

    def calc_weight(self, data):
        cdef datum d
        cdef sfv_t r
        datum_py2native(data, d)
        r = self._handle.calc_weight(d)
        return _to_features(r)

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }

cdef _to_features(sfv_t& r):
    cdef int sz
    sz = r.size()
    ret = [None] * sz
    for i in range(sz):
        ret[i] = Feature(r[i].first.decode('utf8'),
                         r[i].second)
    return ret
