#include <string>

#include <jubatus/core/common/jsonconfig.hpp>
#include <jubatus/core/fv_converter/converter_config.hpp>
#include <jubatus/core/storage/storage_factory.hpp>
#include <jubatus/core/storage/column_table.hpp>
#include <jubatus/core/anomaly/anomaly_factory.hpp>
#include <jubatus/core/classifier/classifier_factory.hpp>
#include <jubatus/core/clustering/clustering.hpp>
#include <jubatus/core/nearest_neighbor/nearest_neighbor_factory.hpp>
#include <jubatus/core/recommender/recommender_factory.hpp>
#include <jubatus/core/regression/regression_factory.hpp>

#include "_wrapper.h"

namespace jsonconfig = jubatus::core::common::jsonconfig;
using jubatus::core::fv_converter::converter_config;
using jubatus::core::fv_converter::make_fv_converter;
using jubatus::core::storage::storage_base;
using jubatus::core::storage::storage_factory;
using jubatus::util::lang::lexical_cast;

void parse_config(const std::string& config, std::string& method,
                  jsonconfig::config& params, converter_config& fvconv_config) {
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

_NearestNeighbor::_NearestNeighbor(const std::string& config) {
    using jubatus::core::storage::column_table;
    using jubatus::core::driver::nearest_neighbor;
    using jubatus::core::nearest_neighbor::create_nearest_neighbor;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    std::string my_id;
    shared_ptr<column_table> table(new column_table);
    handle.reset(new nearest_neighbor(
        create_nearest_neighbor(method, params, table, my_id),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _NearestNeighbor::set_row(const std::string& id, const datum& d) {
    handle->set_row(id, d);
}

id_score_list_t _NearestNeighbor::neighbor_row_from_id(const std::string& id, size_t size) {
    return handle->neighbor_row_from_id(id, size);
}

id_score_list_t _NearestNeighbor::neighbor_row_from_datum(const datum& d, size_t size) {
    return handle->neighbor_row_from_datum(d, size);
}

id_score_list_t _NearestNeighbor::similar_row_from_id(const std::string& id, size_t size) {
    return handle->similar_row(id, size);
}

id_score_list_t _NearestNeighbor::similar_row_from_datum(const datum& d, size_t size) {
    return handle->similar_row(d, size);
}

std::vector<std::string> _NearestNeighbor::get_all_rows() {
    return handle->get_all_rows();
}

_Anomaly::_Anomaly(const std::string& config) : idgen(0) {
    using jubatus::core::driver::anomaly;
    using jubatus::core::anomaly::anomaly_factory;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    std::string my_id;
    handle.reset(new anomaly(
        anomaly_factory::create_anomaly(method, params, my_id),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _Anomaly::load(const std::string& data, const std::string& type, uint64_t version) {
    _Base::load(data, type, version);
    idgen = handle->find_max_int_id() + 1;
}

void _Anomaly::clear_row(const std::string& id) {
    handle->clear_row(id);
}

std::pair<std::string, float> _Anomaly::add(const datum& d) {
    std::string id = lexical_cast<std::string>(idgen++);
    return handle->add(id, d);
}

float _Anomaly::update(const std::string& id, const datum& d) {
    return handle->update(id, d);
}

float _Anomaly::overwrite(const std::string &id, const datum& d) {
    return handle->overwrite(id, d);
}

float _Anomaly::calc_score(const datum& d) const {
    return handle->calc_score(d);
}

std::vector<std::string> _Anomaly::get_all_rows() const {
    return handle->get_all_rows();
}

_Clustering::_Clustering(const std::string& config) {
    using jubatus::core::clustering::clustering;
    using jubatus::core::clustering::clustering_config;
    std::string method;
    jsonconfig::config params;
    converter_config fvconv_config;
    parse_config(config, method, params, fvconv_config);
    std::string my_id;
    clustering_config cluster_conf = jsonconfig::config_cast_check<clustering_config>(params);
    handle.reset(new jubatus::core::driver::clustering(
        shared_ptr<clustering>(
            new clustering(my_id, method, cluster_conf)),
        make_fv_converter(fvconv_config, NULL)));
    this->config.assign(config);
}

void _Clustering::push(const std::vector<datum>& points) {
    handle->push(points);
}

size_t _Clustering::get_revision() const {
    return handle->get_revision();
}

cluster_set _Clustering::get_core_members() const {
    return handle->get_core_members();
}

std::vector<datum> _Clustering::get_k_center() const {
    return handle->get_k_center();
}

datum _Clustering::get_nearest_center(const datum& d) const {
    return handle->get_nearest_center(d);
}

cluster_unit _Clustering::get_nearest_members(const datum& d) const {
    return handle->get_nearest_members(d);
}
