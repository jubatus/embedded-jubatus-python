#include <string>

#include <numpy/ndarrayobject.h>
#include <jubatus/core/fv_converter/datum.hpp>

using jubatus::core::fv_converter::datum;

void allocate_number_string(std::size_t);
const std::string& get_number_string(std::size_t);
const std::string& get_number_string_fast(std::size_t);

void ndarray_to_datum(const PyObject*, std::size_t, datum&);
void csr_to_datum(const PyObject*, const PyObject*, const PyObject*, std::size_t, datum&);
