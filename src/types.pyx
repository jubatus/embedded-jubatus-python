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

cdef datum_native2py(datum& d):
    ret = Datum()
    for i in range(d.string_values_.size()):
        k = d.string_values_[i].first.decode('utf8')
        v = d.string_values_[i].second.decode('utf8')
        ret.add_string(k, v)
    for i in range(d.num_values_.size()):
        k = d.num_values_[i].first.decode('utf8')
        v = d.num_values_[i].second
        ret.add_number(k, v)
    for i in range(d.binary_values_.size()):
        k = d.binary_values_[i].first.decode('utf8')
        v = d.binary_values_[i].second
        ret.add_binary(k, v)
    return ret
