from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map

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
        void push(const vector[datum]& points)
        size_t get_revision()
        vector[vector[pair[double, datum]]] get_core_members()
        vector[datum] get_k_center()
        datum get_nearest_center(const datum& d)
        vector[pair[double, datum]] get_nearest_members(const datum& d)
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
