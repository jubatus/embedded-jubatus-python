cdef _to_window(const pair[double, vector[_Burst.Batch]]& r):
    return Window(r.first, [
        Batch(r.second[i].all_data_count,
              r.second[i].relevant_data_count,
              r.second[i].burst_weight)
        for i in range(r.second.size())
    ])

cdef _to_window_list(map[string, pair[double, vector[_Burst.Batch]]]& r):
    return {it.first.decode('utf8'): _to_window(it.second) for it in r}

cdef class Burst(_JubatusBase):
    cdef _Burst *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Burst(config)
        self._type, self._model_ver = b'burst', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def add_documents(self, data):
        ret = 0
        for doc in data:
            if self._handle.add_document(doc.text.encode('utf8'), doc.pos):
                ret += 1
        if ret > 0:
            self._handle.calculate_results()
        return ret

    def get_result(self, keyword):
        return _to_window(self._handle.get_result(keyword.encode('utf8')))

    def get_result_at(self, keyword, pos):
        return _to_window(self._handle.get_result_at(keyword.encode('utf8'), pos))

    def get_all_bursted_results(self):
        cdef map[string, pair[double, vector[_Burst.Batch]]] r
        r = self._handle.get_all_bursted_results()
        return _to_window_list(r)

    def get_all_bursted_results_at(self, pos):
        cdef map[string, pair[double, vector[_Burst.Batch]]] r
        r = self._handle.get_all_bursted_results_at(pos)
        return _to_window_list(r)

    def get_all_keywords(self):
        cdef vector[keyword_with_params] r
        r = self._handle.get_all_keywords()
        return [
            KeywordWithParams(r[i].keyword.decode('utf8'), r[i].scaling_param, r[i].gamma)
            for i in range(r.size())
        ]

    def add_keyword(self, keyword):
        cdef keyword_params kp
        kp.scaling_param = keyword.scaling_param
        kp.gamma = keyword.gamma
        return self._handle.add_keyword(keyword.keyword.encode('utf8'), kp)

    def remove_keyword(self, keyword):
        return self._handle.remove_keyword(keyword.encode('utf8'))

    def remove_all_keywords(self):
        return self._handle.remove_all_keywords()

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
