cdef class Graph(_JubatusBase):
    cdef _Graph *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Graph(config)
        self._type, self._model_ver = b'graph', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def create_node(self):
        return self._handle.create_node().decode('ascii')

    def remove_node(self, node_id):
        return self._handle.remove_node(node_id.encode('ascii'))

    def update_node(self, node_id, props):
        cdef prop_t p
        props_py2native(props, p)
        return self._handle.update_node(node_id.encode('ascii'), p)

    def create_edge(self, node_id, e):
        if not isinstance(e, Edge):
            raise ValueError
        if node_id != e.source:
            raise ValueError
        cdef prop_t p
        props_py2native(e.property, p)
        return self._handle.create_edge(node_id.encode('ascii'),
                                        e.target.encode('ascii'), p)

    def update_edge(self, node_id, edge_id, e):
        if not isinstance(e, Edge):
            raise ValueError
        if node_id != e.source:
            raise ValueError
        cdef prop_t p
        props_py2native(e.property, p)
        return self._handle.update_edge(edge_id, p)

    def remove_edge(self, node_id, edge_id):
        self._handle.remove_edge(edge_id)

    def get_centrality(self, node_id, centrality_type, query):
        if not isinstance(query, PresetQuery):
            raise ValueError
        cdef preset_query q
        preset_query_py2native(query, q)
        return self._handle.get_centrality(node_id.encode('ascii'),
                                           centrality_type, q)

    def add_centrality_query(self, query):
        if not isinstance(query, PresetQuery):
            raise ValueError
        cdef preset_query q
        preset_query_py2native(query, q)
        self._handle.add_centrality_query(q)
        return True

    def add_shortest_path_query(self, query):
        if not isinstance(query, PresetQuery):
            raise ValueError
        cdef preset_query q
        preset_query_py2native(query, q)
        self._handle.add_shortest_path_query(q)
        return True

    def remove_centrality_query(self, query):
        if not isinstance(query, PresetQuery):
            raise ValueError
        cdef preset_query q
        preset_query_py2native(query, q)
        self._handle.remove_centrality_query(q)
        return True

    def remove_shortest_path_query(self, query):
        if not isinstance(query, PresetQuery):
            raise ValueError
        cdef preset_query q
        preset_query_py2native(query, q)
        self._handle.remove_shortest_path_query(q)
        return True

    def get_shortest_path(self, query):
        if not isinstance(query, ShortestPathQuery):
            raise ValueError
        cdef preset_query q
        cdef vector[node_id_t] r
        preset_query_py2native(query.query, q)
        r = self._handle.get_shortest_path(query.source.encode('ascii'),
                                           query.target.encode('ascii'),
                                           query.max_hop, q)
        return [str(x) for x in r]

    def update_index(self):
        self._handle.update_index()
        return True

    def get_node(self, node_id):
        cdef node_info n
        n = self._handle.get_node(node_id.encode('ascii'))
        return Node(props_native2py(n.property),
                    edges_native2py(n.in_edges),
                    edges_native2py(n.out_edges))

    def get_edge(self, node_id, edge_id):
        cdef edge_info e
        e = self._handle.get_edge(edge_id)
        return Edge(props_native2py(e.p),
                    str(e.src), str(e.tgt))

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
