from _wrapper cimport datum
from jubatus.common.datum import Datum

cdef datum_py2native(pd, datum& d):
    d.string_values_.clear()
    for k, v in pd.string_values:
        k = k.encode('utf8')
        v = v.encode('utf8')
        d.string_values_.push_back(<tuple>(k, v))
    d.num_values_.clear()
    for k, v in pd.num_values:
        k = k.encode('utf8')
        d.num_values_.push_back(<tuple>(k, v))
    d.binary_values_.clear()
    for k, v in pd.binary_values:
        k = k.encode('utf8')
        d.binary_values_.push_back(<tuple>(k, v))

def datum_native2py(d):
    pass
