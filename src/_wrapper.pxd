from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map

ctypedef vector[pair[string, float]] sfv_t
ctypedef map[string, string] prop_t
ctypedef uint64_t edge_id_t
ctypedef uint64_t node_id_t

cdef extern from '_wrapper.h' nogil:
    cdef cppclass _Classifier:
        _Classifier(const string& config) except +
        void train(const string& config, const datum& d)
        vector[classify_result_elem] classify(const datum& d)
        vector[pair[string, uint64_t]] get_labels()
        bool set_label(const string& new_label)
        bool delete_label(const string& target_label)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Regression:
        _Regression(const string& config) except +
        void train(float score, const datum& d)
        float estimate(const datum& d)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Recommender:
        _Recommender(const string& config) except +
        void clear_row(const string& id)
        void update_row(const string& id, const datum& d)
        datum complete_row_from_id(const string& id)
        datum complete_row_from_datum(const datum& d)
        vector[pair[string, float]] similar_row_from_id(const string& id, size_t ret_num)
        vector[pair[string, float]] similar_row_from_datum(const datum& d, size_t ret_num)
        datum decode_row(const string& id)
        vector[string] get_all_rows()
        float calc_similarity(const datum& l, const datum& r)
        float calc_l2norm(const datum& d)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _NearestNeighbor:
        _NearestNeighbor(const string& config) except +
        void set_row(const string& id, const datum& d)
        vector[pair[string, float]] neighbor_row_from_id(const string& id, size_t size)
        vector[pair[string, float]] neighbor_row_from_datum(const datum& d, size_t size)
        vector[pair[string, float]] similar_row_from_id(const string& id, size_t size)
        vector[pair[string, float]] similar_row_from_datum(const datum& d, size_t size)
        vector[string] get_all_rows()
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Anomaly:
        _Anomaly(const string& config) except +
        void clear_row(const string& id)
        pair[string, float] add(const datum& d)
        float update(const string& id, const datum& d)
        float overwrite(const string& id, const datum& d)
        float calc_score(const datum& d)
        vector[string] get_all_rows()
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Clustering:
        _Clustering(const string& config) except +
        void push(const vector[indexed_point]& points)
        size_t get_revision()
        vector[vector[pair[double, datum]]] get_core_members()
        vector[datum] get_k_center()
        datum get_nearest_center(const datum& d)
        vector[pair[double, datum]] get_nearest_members(const datum& d)
        vector[vector[pair[double, string]]] get_core_members_light()
        vector[pair[double, string]] get_nearest_members_light(const datum& d)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Burst:
        cppclass Batch:
            int all_data_count
            int relevant_data_count
            double burst_weight
        _Burst(const string& config) except +
        bool add_document(const string& str, double pos)
        pair[double,vector[Batch]] get_result(const string& keyword)
        pair[double,vector[Batch]] get_result_at(const string& keyword, double pos)
        map[string, pair[double,vector[Batch]]] get_all_bursted_results()
        map[string, pair[double,vector[Batch]]] get_all_bursted_results_at(double pos)
        vector[keyword_with_params] get_all_keywords()
        bool add_keyword(const string& keyword, const keyword_params& params)
        bool remove_keyword(const string& keyword)
        bool remove_all_keywords()
        void calculate_results()
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Bandit:
        _Bandit(const string& config) except +
        bool register_arm(const string& arm_id)
        bool delete_arm(const string& arm_id)
        string select_arm(const string& player_id)
        bool register_reward(const string& player_id, const string& arm_id, double reward)
        map[string, arm_info] get_arm_info(const string& player_id)
        bool reset(const string& player_id)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Stat:
        _Stat(const string& config) except +
        void push(const string& key, double value)
        double sum(const string& key)
        double stddev(const string& key)
        double max(const string& key)
        double min(const string& key)
        double entropy()
        double moment(const string& key, int degree, double center)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Weight:
        _Weight(const string& config) except +
        sfv_t update(const datum& d)
        sfv_t calc_weight(const datum& d)
        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

    cdef cppclass _Graph:
        _Graph(const string& config) except +
        string create_node()
        bool remove_node(const string& node_id)
        bool update_node(const string& node_id,
                         const prop_t& properties)
        edge_id_t create_edge(const string& src,
                              const string& target,
                              const prop_t& properties)
        bool update_edge(edge_id_t edge_id,
                         const prop_t& properties)
        void remove_edge(edge_id_t edge_id)

        double get_centrality(const string& node_id,
                              int centrality_type,
                              const preset_query& q)
        void add_centrality_query(const preset_query& q)
        void add_shortest_path_query(const preset_query& q)
        void remove_centrality_query(const preset_query& q)
        void remove_shortest_path_query(const preset_query& q)
        vector[node_id_t] get_shortest_path(const string& src,
                                            const string& target,
                                            uint64_t max_hop,
                                            const preset_query& q)

        void update_index()
        node_info get_node(const string& node_id)
        edge_info get_edge(edge_id_t eid)

        string dump(const string& type, uint64_t ver)
        void load(const string& data, const string& type, uint64_t ver)
        string get_config()
        void clear()

cdef extern from 'jubatus/core/fv_converter/datum.hpp' namespace 'jubatus::core::fv_converter' nogil:
    cdef cppclass datum:
        vector[pair[string, string]] string_values_
        vector[pair[string, double]] num_values_
        vector[pair[string, string]] binary_values_

cdef extern from 'jubatus/core/classifier/classifier_type.hpp' namespace 'jubatus::core::classifier' nogil:
    cdef cppclass classify_result_elem:
        string label
        float score

cdef extern from 'jubatus/core/burst/burst.hpp' namespace 'jubatus::core::burst' nogil:
    cdef cppclass keyword_params:
        double scaling_param
        double gamma

    cdef cppclass keyword_with_params:
        string keyword
        double scaling_param
        double gamma

cdef extern from 'jubatus/core/bandit/arm_info.hpp' namespace 'jubatus::core::bandit' nogil:
    cdef cppclass arm_info:
        int trial_count
        double weight

cdef extern from 'jubatus/util/lang/cast.h' namespace 'jubatus::util::lang' nogil:
    cdef T lexical_cast[T, S](const S& arg)

cdef extern from 'jubatus/core/graph/graph_type.hpp' namespace 'jubatus::core::graph' nogil:
    cdef cppclass node_info:
        prop_t property
        vector[edge_id_t] in_edges
        vector[edge_id_t] out_edges

    cdef cppclass edge_info:
        prop_t p
        node_id_t src
        node_id_t tgt

    cdef cppclass preset_query:
        vector[pair[string, string]] edge_query
        vector[pair[string, string]] node_query

cdef extern from 'jubatus/core/clustering/types.hpp' namespace 'jubatus::core::clustering' nogil:
    cdef cppclass indexed_point:
        string id
        datum point
