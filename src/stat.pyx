cdef class _StatWrapper:
    cdef _Stat *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Stat(config)
        typ, ver = b'stat', 1
        return (
            lambda: self._handle.get_config().decode('utf8'),
            lambda: self._handle.dump(typ, ver),
            lambda x: self._handle.load(x, typ, ver),
            lambda: self._handle.clear(),
            typ,
        )

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
