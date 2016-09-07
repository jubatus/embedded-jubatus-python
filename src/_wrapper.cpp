#include <string>

#include <jubatus/core/common/jsonconfig.hpp>
#include <jubatus/core/fv_converter/converter_config.hpp>
#include <jubatus/core/storage/storage_factory.hpp>
#include <jubatus/core/classifier/classifier_factory.hpp>

#include "_wrapper.h"

namespace jsonconfig = jubatus::core::common::jsonconfig;
using jubatus::core::fv_converter::converter_config;
using jubatus::core::fv_converter::make_fv_converter;

void parse_config(const std::string& config, std::string& method,
                  jsonconfig::config& params, converter_config& fvconv_config) {
    using jubatus::util::lang::lexical_cast;
    using jubatus::util::text::json::json;
    using jubatus::util::text::json::json_string;
    using jubatus::util::text::json::from_json;
    json config_json = lexical_cast<json>(config);
    json_string *method_value = (json_string*)config_json["method"].get();
    if (!method_value || method_value->type() != json::String)
        throw std::invalid_argument("invalid config (method)");
    method.assign(method_value->get());
    from_json(config_json["converter"], fvconv_config);
    params = jsonconfig::config(config_json["parameter"]);
}

_Classifier::_Classifier(const std::string& config) {
    using jubatus::core::classifier::classifier_factory;
    using jubatus::core::driver::classifier;
    using jubatus::core::storage::storage_factory;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    handle.reset(new classifier(classifier_factory::create_classifier(
        method, params, storage_factory::create_storage("local")),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _Classifier::train(const std::string& label, const datum& d) {
    handle->train(label, d);
}

classify_result _Classifier::classify(const datum& d) {
    return handle->classify(d);
}

std::vector<std::pair<std::string, uint64_t> > _Classifier::get_labels() {
    using jubatus::core::classifier::labels_t;
    std::vector<std::pair<std::string, uint64_t> > ret;
    labels_t tmp = handle->get_labels();
    for (labels_t::iterator it = tmp.begin(); it != tmp.end(); ++it) {
        ret.push_back(std::pair<std::string, uint64_t>(it->first, it->second));
    }
    return ret;
}

bool _Classifier::set_label(const std::string& new_label) {
    return handle->set_label(new_label);
}

bool _Classifier::delete_label(const std::string& target_label) {
    return handle->delete_label(target_label);
}
