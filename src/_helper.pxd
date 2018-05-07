from cpython.ref cimport PyObject
from libcpp.string cimport string

from _wrapper cimport datum

cdef extern from '_helper.h':
    cdef void ndarray_to_datum(const PyObject* obj, size_t i, datum&) except +
    cdef void csr_to_datum(const PyObject*, const PyObject*, const PyObject*, size_t, datum&) except +

cdef extern from '_helper.h' nogil:
    cdef void allocate_number_string(size_t max_num) except +
    cdef const string& get_number_string(size_t num) except +
    cdef const string& get_number_string_fast(size_t num) except +
