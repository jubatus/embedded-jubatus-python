cdef class Recommender(_JubatusBase):
    cdef _Recommender *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Recommender(config)
        self._type, self._model_ver = b'recommender', 1

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

    def update_row(self, id_, row):
        cdef datum d
        datum_py2native(row, d)
        self._handle.update_row(id_.encode('utf8'), d)
        return True

    def complete_row_from_id(self, id_):
        cdef datum d = self._handle.complete_row_from_id(id_.encode('utf8'))
        return datum_native2py(d)

    def complete_row_from_datum(self, row):
        cdef datum d0
        datum_py2native(row, d0)
        cdef datum d1 = self._handle.complete_row_from_datum(d0)
        return datum_native2py(d1)

    def similar_row_from_id(self, id_, size):
        cdef vector[pair[string, double]] ret
        ret = self._handle.similar_row_from_id(id_.encode('utf8'), size)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_id_and_score(self, id_, score):
        cdef vector[pair[string, double]] ret
        ret = self._handle.similar_row_from_id_and_score(id_.encode('utf8'), score)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_id_and_rate(self, id_, rate):
        cdef vector[pair[string, double]] ret
        ret = self._handle.similar_row_from_id_and_rate(id_.encode('utf8'), rate)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_datum(self, row, size):
        cdef vector[pair[string, double]] ret
        cdef datum d
        datum_py2native(row, d)
        ret = self._handle.similar_row_from_datum(d, size)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_datum_and_score(self, row, score):
        cdef vector[pair[string, double]] ret
        cdef datum d
        datum_py2native(row, d)
        ret = self._handle.similar_row_from_datum_and_score(d, score)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def similar_row_from_datum_and_rate(self, row, rate):
        cdef vector[pair[string, double]] ret
        cdef datum d
        datum_py2native(row, d)
        ret = self._handle.similar_row_from_datum_and_rate(d, rate)
        return [
            RecommenderIdWithScore(ret[i].first.decode('utf8'), ret[i].second)
            for i in range(ret.size())
        ]

    def decode_row(self, id_):
        cdef datum d = self._handle.decode_row(id_.encode('utf8'))
        return datum_native2py(d)

    def get_all_rows(self):
        cdef vector[string] ret = self._handle.get_all_rows()
        return [(<string>ret[i]).decode('utf8') for i in range(ret.size())]

    def calc_similarity(self, l, r):
        cdef datum d0
        cdef datum d1
        datum_py2native(l, d0)
        datum_py2native(r, d1)
        return self._handle.calc_similarity(d0, d1)

    def calc_l2norm(self, row):
        cdef datum d
        datum_py2native(row, d)
        return self._handle.calc_l2norm(d)

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
