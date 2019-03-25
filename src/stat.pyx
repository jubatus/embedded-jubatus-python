cdef class Stat(_JubatusBase):
    cdef _Stat *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Stat(config)
        self._type, self._model_ver = b'stat', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def push(self, key, value):
        self._handle.push(key.encode('utf8'), value)
        return True

    def sum(self, key):
        return self._handle.sum(key.encode('utf8'))

    def stddev(self, key):
        return self._handle.stddev(key.encode('utf8'))

    def max(self, key):
        return self._handle.max(key.encode('utf8'))

    def min(self, key):
        return self._handle.min(key.encode('utf8'))

    def entropy(self, key):
        return self._handle.entropy()

    def moment(self, key, degree, center):
        return self._handle.moment(key.encode('utf8'), degree, center)

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
