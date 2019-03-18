from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map

ctypedef vector[pair[string, double]] sfv_t
ctypedef map[string, map[string, string]] status_t
ctypedef map[string, string] prop_t
ctypedef uint64_t edge_id_t
ctypedef uint64_t node_id_t

cdef extern from '_wrapper.h' nogil:
    cdef cppclass _Classifier:
        _Classifier(const string& config) except +
        void train(const string& config, const datum& d) except +
        vector[classify_result_elem] classify(const datum& d) except +
        vector[pair[string, uint64_t]] get_labels() except +
        bool set_label(const string& new_label) except +
        bool delete_label(const string& target_label) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Regression:
        _Regression(const string& config) except +
        void train(double score, const datum& d) except +
        double estimate(const datum& d) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Recommender:
        _Recommender(const string& config) except +
        void clear_row(const string& id) except +
        void update_row(const string& id, const datum& d) except +
        datum complete_row_from_id(const string& id) except +
        datum complete_row_from_datum(const datum& d) except +
        vector[pair[string, double]] similar_row_from_id(const string& id, size_t ret_num) except +
        vector[pair[string, double]] similar_row_from_id_and_score(const string& id, double score) except +
        vector[pair[string, double]] similar_row_from_id_and_rate(const string& id, float rate) except +
        vector[pair[string, double]] similar_row_from_datum(const datum& d, size_t ret_num) except +
        vector[pair[string, double]] similar_row_from_datum_and_score(const datum& d, double score) except +
        vector[pair[string, double]] similar_row_from_datum_and_rate(const datum& d, float rate) except +
        datum decode_row(const string& id) except +
        vector[string] get_all_rows() except +
        double calc_similarity(const datum& l, const datum& r) except +
        double calc_l2norm(const datum& d) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _NearestNeighbor:
        _NearestNeighbor(const string& config) except +
        void set_row(const string& id, const datum& d) except +
        vector[pair[string, double]] neighbor_row_from_id(const string& id, size_t size) except +
        vector[pair[string, double]] neighbor_row_from_datum(const datum& d, size_t size) except +
        vector[pair[string, double]] similar_row_from_id(const string& id, size_t size) except +
        vector[pair[string, double]] similar_row_from_datum(const datum& d, size_t size) except +
        vector[string] get_all_rows() except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Anomaly:
        _Anomaly(const string& config) except +
        void clear_row(const string& id) except +
        pair[string, double] add(const datum& d) except +
        vector[string] add_bulk(const vector[pair[string, datum]]& data) except +
        double update(const string& id, const datum& d) except +
        double overwrite(const string& id, const datum& d) except +
        double calc_score(const datum& d) except +
        vector[string] get_all_rows() except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        string get_next_id() except +
        status_t get_status() except +

    cdef cppclass _Clustering:
        _Clustering(const string& config) except +
        void push(const vector[indexed_point]& points) except +
        size_t get_revision() except +
        vector[vector[pair[double, datum]]] get_core_members() except +
        vector[datum] get_k_center() except +
        datum get_nearest_center(const datum& d) except +
        vector[pair[double, datum]] get_nearest_members(const datum& d) except +
        vector[vector[pair[double, string]]] get_core_members_light() except +
        vector[pair[double, string]] get_nearest_members_light(const datum& d) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Burst:
        cppclass Batch:
            int all_data_count
            int relevant_data_count
            double burst_weight
        _Burst(const string& config) except +
        bool add_document(const string& str, double pos) except +
        pair[double,vector[Batch]] get_result(const string& keyword) except +
        pair[double,vector[Batch]] get_result_at(const string& keyword, double pos) except +
        map[string, pair[double,vector[Batch]]] get_all_bursted_results() except +
        map[string, pair[double,vector[Batch]]] get_all_bursted_results_at(double pos) except +
        vector[keyword_with_params] get_all_keywords() except +
        bool add_keyword(const string& keyword, const keyword_params& params) except +
        bool remove_keyword(const string& keyword) except +
        bool remove_all_keywords() except +
        void calculate_results() except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Bandit:
        _Bandit(const string& config) except +
        bool register_arm(const string& arm_id) except +
        bool delete_arm(const string& arm_id) except +
        string select_arm(const string& player_id) except +
        bool register_reward(const string& player_id, const string& arm_id, double reward) except +
        map[string, arm_info] get_arm_info(const string& player_id) except +
        bool reset(const string& player_id) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Stat:
        _Stat(const string& config) except +
        void push(const string& key, double value) except +
        double sum(const string& key) except +
        double stddev(const string& key) except +
        double max(const string& key) except +
        double min(const string& key) except +
        double entropy() except +
        double moment(const string& key, int degree, double center) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Weight:
        _Weight(const string& config) except +
        sfv_t update(const datum& d) except +
        sfv_t calc_weight(const datum& d) except +
        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

    cdef cppclass _Graph:
        _Graph(const string& config) except +
        string create_node() except +
        bool remove_node(const string& node_id) except +
        bool update_node(const string& node_id,
                         const prop_t& properties) except +
        edge_id_t create_edge(const string& src,
                              const string& target,
                              const prop_t& properties) except +
        bool update_edge(edge_id_t edge_id,
                         const prop_t& properties) except +
        void remove_edge(edge_id_t edge_id) except +

        double get_centrality(const string& node_id,
                              int centrality_type,
                              const preset_query& q) except +
        void add_centrality_query(const preset_query& q) except +
        void add_shortest_path_query(const preset_query& q) except +
        void remove_centrality_query(const preset_query& q) except +
        void remove_shortest_path_query(const preset_query& q) except +
        vector[node_id_t] get_shortest_path(const string& src,
                                            const string& target,
                                            uint64_t max_hop,
                                            const preset_query& q) except +

        void update_index() except +
        node_info get_node(const string& node_id) except +
        edge_info get_edge(edge_id_t eid) except +

        string dump(const string& type, uint64_t ver) except +
        void load(const string& data, const string& type, uint64_t ver) except +
        string get_config() except +
        void clear() except +
        status_t get_status() except +

cdef extern from 'jubatus/core/fv_converter/datum.hpp' namespace 'jubatus::core::fv_converter' nogil:
    cdef cppclass datum:
        vector[pair[string, string]] string_values_
        vector[pair[string, double]] num_values_
        vector[pair[string, string]] binary_values_

cdef extern from 'jubatus/core/classifier/classifier_type.hpp' namespace 'jubatus::core::classifier' nogil:
    cdef cppclass classify_result_elem:
        string label
        double score

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
