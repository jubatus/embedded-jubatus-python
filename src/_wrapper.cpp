#include <string>

#include <jubatus/core/common/jsonconfig.hpp>
#include <jubatus/core/fv_converter/converter_config.hpp>
#include <jubatus/core/storage/storage_factory.hpp>
#include <jubatus/core/classifier/classifier_factory.hpp>
#include <jubatus/core/recommender/recommender_factory.hpp>
#include <jubatus/core/regression/regression_factory.hpp>

#include "_wrapper.h"

namespace jsonconfig = jubatus::core::common::jsonconfig;
using jubatus::core::fv_converter::converter_config;
using jubatus::core::fv_converter::make_fv_converter;
using jubatus::core::storage::storage_base;
using jubatus::core::storage::storage_factory;

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

_Regression::_Regression(const std::string& config) {
    using jubatus::core::driver::regression;
    using jubatus::core::regression::regression_factory;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    shared_ptr<storage_base> model = storage_factory::create_storage("local");
    handle.reset(new regression(model,
        regression_factory::create_regression(method, params, model),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _Regression::train(float score, const datum& d) {
    handle->train(std::pair<float, datum>(score, d));
}

float _Regression::estimate(const datum& d) {
    return handle->estimate(d);
}

_Recommender::_Recommender(const std::string& config) {
    using jubatus::core::driver::recommender;
    using jubatus::core::recommender::recommender_factory;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    std::string my_id;
    handle.reset(new recommender(
        recommender_factory::create_recommender(method, params, my_id),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _Recommender::clear_row(const std::string& id) {
    handle->clear_row(id);
}

void _Recommender::update_row(const std::string& id, const datum& d) {
    handle->update_row(id, d);
}

datum _Recommender::complete_row_from_id(const std::string& id) {
    return handle->complete_row_from_id(id);
}

datum _Recommender::complete_row_from_datum(const datum& d) {
    return handle->complete_row_from_datum(d);
}

std::vector<std::pair<std::string, float> > _Recommender::similar_row_from_id(const std::string& id, size_t ret_num) {
    return handle->similar_row_from_id(id, ret_num);
}

std::vector<std::pair<std::string, float> > _Recommender::similar_row_from_datum(const datum& d, size_t ret_num) {
    return handle->similar_row_from_datum(d, ret_num);
}

datum _Recommender::decode_row(const std::string& id) {
    return handle->decode_row(id);
}

std::vector<std::string> _Recommender::get_all_rows() {
    return handle->get_all_rows();
}

float _Recommender::calc_similarity(const datum& l, const datum& r) {
    return handle->calc_similarity(l, r);
}

float _Recommender::calc_l2norm(const datum& d) {
    return handle->calc_l2norm(d);
}
