cdef class Clustering(_JubatusBase):
    cdef _Clustering *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Clustering(config)
        self._type, self._model_ver = b'clustering', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def push(self, points):
        cdef datum d
        cdef vector[indexed_point] v
        cdef indexed_point ip
        for p in points:
            datum_py2native(p.point, d)
            ip.id = p.id.encode('utf8')
            ip.point = d
            v.push_back(ip)
        self._handle.push(v)
        return True

    def get_revision(self):
        return self._handle.get_revision()

    def get_core_members(self):
        cdef vector[vector[pair[double, datum]]] r
        r = self._handle.get_core_members()
        ret = []
        for i in range(r.size()):
            ret.append([
                WeightedDatum(r[i][j].first, datum_native2py(r[i][j].second))
                for j in range(r[i].size())
            ])
        return ret

    def get_k_center(self):
        cdef vector[datum] r = self._handle.get_k_center()
        return [datum_native2py(r[i]) for i in range(r.size())]

    def get_nearest_center(self, point):
        cdef datum d
        cdef datum r
        datum_py2native(point, d)
        r = self._handle.get_nearest_center(d)
        return datum_native2py(r)

    def get_nearest_members(self, point):
        cdef datum d
        cdef vector[pair[double, datum]] r
        datum_py2native(point, d)
        r = self._handle.get_nearest_members(d)
        return [
            WeightedDatum(r[i].first, datum_native2py(r[i].second))
            for i in range(r.size())
        ]

    def get_core_members_light(self):
        cdef vector[vector[pair[double, string]]] r
        r = self._handle.get_core_members_light()
        ret = []
        for c in r:
            tmp = []
            for m in c:
                tmp.append(WeightedIndex(m.first, m.second))
            ret.append(tmp)
        return ret

    def get_nearest_members_light(self, point):
        cdef vector[pair[double, string]] r
        cdef datum d
        datum_py2native(point, d)
        r = self._handle.get_nearest_members_light(d)
        ret = []
        for m in r:
            ret.append(WeightedIndex(m.first, m.second))
        return ret

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
