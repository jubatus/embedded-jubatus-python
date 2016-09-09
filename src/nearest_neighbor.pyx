cdef class _NearestNeighborWrapper:
    cdef _NearestNeighbor *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _NearestNeighbor(config)
        typ, ver = b'nearest_neighbor', 1
        return (
            lambda: self._handle.get_config().decode('utf8'),
            lambda: self._handle.dump(typ, ver),
            lambda x: self._handle.load(x, typ, ver),
            lambda: self._handle.clear(),
            typ,
        )

    def set_row(self, id_, row):
        cdef datum d
        datum_py2native(row, d)
        self._handle.set_row(id_.encode('utf8'), d)
        return True

    def neighbor_row_from_id(self, id_, size):
        cdef vector[pair[string, float]] ret
        ret = self._handle.neighbor_row_from_id(id_.encode('utf8'), size)
        return [
            NNIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def neighbor_row_from_datum(self, dat, size):
        cdef datum d
        cdef vector[pair[string, float]] ret
        datum_py2native(dat, d)
        ret = self._handle.neighbor_row_from_datum(d, size)
        return [
            NNIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_id(self, id_, size):
        cdef vector[pair[string, float]] ret
        ret = self._handle.similar_row_from_id(id_.encode('utf8'), size)
        return [
            NNIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_datum(self, dat, size):
        cdef datum d
        cdef vector[pair[string, float]] ret
        datum_py2native(dat, d)
        ret = self._handle.similar_row_from_datum(d, size)
        return [
            NNIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def get_all_rows(self):
        cdef vector[string] ret = self._handle.get_all_rows()
        return [(<string>ret[i]).decode('utf8') for i in range(ret.size())]
