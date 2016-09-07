from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from '_wrapper.h':
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

cdef extern from 'jubatus/core/fv_converter/datum.hpp' namespace 'jubatus::core::fv_converter':
    cdef cppclass datum:
        vector[pair[string, string]] string_values_
        vector[pair[string, double]] num_values_
        vector[pair[string, string]] binary_values_

cdef extern from 'jubatus/core/classifier/classifier_type.hpp' namespace 'jubatus::core::classifier':
    cdef cppclass classify_result_elem:
        string label
        float score
