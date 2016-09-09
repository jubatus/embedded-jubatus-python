cdef class _RegressionWrapper:
    cdef _Regression *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Regression(config)
        typ, ver = b'regression', 1
        return (
            lambda: self._handle.get_config().decode('utf8'),
            lambda: self._handle.dump(typ, ver),
            lambda x: self._handle.load(x, typ, ver),
            lambda: self._handle.clear(),
            typ,
        )

    def train(self, data):
        cdef datum d
        if isinstance(data, list):
            if len(data) == 0:
                return 0
            if isinstance(data[0], ScoredDatum):
                for r in data:
                    score = float(r.score)
                    datum_py2native(r.data, d)
                    self._handle.train(score, d)
            elif isinstance(data[0], (tuple, list)):
                for score, datum in data:
                    datum_py2native(datum, d)
                    self._handle.train(float(score), d)
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
