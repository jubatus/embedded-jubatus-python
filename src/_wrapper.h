#include <string>

#include <jubatus/util/lang/shared_ptr.h>
#include <jubatus/util/text/json.h>
#include <jubatus/core/fv_converter/datum.hpp>
#include <jubatus/core/framework/stream_writer.hpp>
#include <jubatus/core/driver/anomaly.hpp>
#include <jubatus/core/driver/bandit.hpp>
#include <jubatus/core/driver/burst.hpp>
#include <jubatus/core/driver/classifier.hpp>
#include <jubatus/core/driver/clustering.hpp>
#include <jubatus/core/driver/nearest_neighbor.hpp>
#include <jubatus/core/driver/recommender.hpp>
#include <jubatus/core/driver/regression.hpp>

using jubatus::util::lang::shared_ptr;
using jubatus::core::fv_converter::datum;
using jubatus::core::classifier::classify_result;
using jubatus::core::clustering::cluster_unit;
using jubatus::core::clustering::cluster_set;

std::string pack_model(const std::string& type,
                       const std::string& config,
                       const std::string& id,
                       const msgpack::sbuffer& user_data_buf);
void unpack_model(const std::string& data,
                  msgpack::unpacked& user_data_buffer,
                  std::string& model_type,
                  std::string& model_id,
                  std::string& model_config,
                  uint64_t *user_data_version,
                  msgpack::object **user_data);

template<typename T>
class _Base {
protected:
    shared_ptr<T> handle;
    std::string config;
public:
    _Base(): handle(), config() {}
    virtual ~_Base() {
        handle.reset();
    }

    std::string get_config() const {
        return config;
    }

    std::string dump(const std::string& type, uint64_t version) const {
        using jubatus::core::framework::stream_writer;
        using jubatus::core::framework::jubatus_packer;
        using jubatus::core::framework::packer;
        static const std::string ID("");
        msgpack::sbuffer user_data_buf;
        {
            stream_writer<msgpack::sbuffer> st(user_data_buf);
            jubatus_packer jp(st);
            packer packer(jp);
            packer.pack_array(2);
            packer.pack(version);
            this->handle->pack(packer);
        }
        return pack_model(type, config, ID, user_data_buf);
    }

    virtual void load(const std::string& data, const std::string& type, uint64_t version) {
        msgpack::unpacked unpacked;
        uint64_t user_data_version;
        msgpack::object *user_data;
        std::string model_type, model_id, model_config;
        unpack_model(data, unpacked, model_type, model_id, model_config, &user_data_version, &user_data);
        if (model_type != type || user_data_version != version)
            throw std::runtime_error("invalid model type or version");
        this->handle->unpack(*user_data);
        this->config.assign(model_config);
    }

    void clear() {
        this->handle->clear();
    }
};

typedef std::vector<std::pair<std::string, float> > id_score_list_t;

class _Classifier : public _Base<jubatus::core::driver::classifier> {
public:
    _Classifier(const std::string& config);
    ~_Classifier() {}
    void train(const std::string& label, const datum& d);
    classify_result classify(const datum& d);
    std::vector<std::pair<std::string, uint64_t> > get_labels();
    bool set_label(const std::string& new_label);
    bool delete_label(const std::string& target_label);
};

class _Regression : public _Base<jubatus::core::driver::regression> {
public:
    _Regression(const std::string& config);
    ~_Regression() {}
    void train(float score, const datum& d);
    float estimate(const datum& d);
};

class _Recommender : public _Base<jubatus::core::driver::recommender> {
public:
    _Recommender(const std::string& config);
    ~_Recommender() {}
    void clear_row(const std::string& id);
    void update_row(const std::string& id, const datum& d);
    datum complete_row_from_id(const std::string& id);
    datum complete_row_from_datum(const datum& d);
    id_score_list_t similar_row_from_id(const std::string& id, size_t ret_num);
    id_score_list_t similar_row_from_datum(const datum& d, size_t ret_num);
    datum decode_row(const std::string& id);
    std::vector<std::string> get_all_rows();
    float calc_similarity(const datum& l, const datum& r);
    float calc_l2norm(const datum& d);
};

class _NearestNeighbor : public _Base<jubatus::core::driver::nearest_neighbor> {
public:
    _NearestNeighbor(const std::string& config);
    ~_NearestNeighbor() {}
    void set_row(const std::string& id, const datum& d);
    id_score_list_t neighbor_row_from_id(const std::string& id, size_t size);
    id_score_list_t neighbor_row_from_datum(const datum& d, size_t size);
    id_score_list_t similar_row_from_id(const std::string& id, size_t size);
    id_score_list_t similar_row_from_datum(const datum& d, size_t size);
    std::vector<std::string> get_all_rows();
};

class _Anomaly : public _Base<jubatus::core::driver::anomaly> {
    uint64_t idgen;
public:
    _Anomaly(const std::string& config);
    ~_Anomaly() {}
    void load(const std::string& data, const std::string& type, uint64_t version);

    void clear_row(const std::string& id);
    std::pair<std::string, float> add(const datum& d);
    float update(const std::string& id, const datum& d);
    float overwrite(const std::string &id, const datum& d);
    float calc_score(const datum& d) const;
    std::vector<std::string> get_all_rows() const;
};

class _Clustering : public _Base<jubatus::core::driver::clustering> {
public:
    _Clustering(const std::string& config);
    ~_Clustering() {}
    void push(const std::vector<datum>& points);
    size_t get_revision() const;
    cluster_set get_core_members() const;
    std::vector<datum> get_k_center() const;
    datum get_nearest_center(const datum& d) const;
    cluster_unit get_nearest_members(const datum& d) const;
};

class _Burst : public _Base<jubatus::core::driver::burst> {
public:
    typedef jubatus::core::burst::keyword_params keyword_params;
    typedef jubatus::core::burst::burst::keyword_list keyword_list;
    struct Batch {
        int all_data_count;
        int relevant_data_count;
        double burst_weight;
    };
    typedef std::pair<double, std::vector<Batch> > window;
    typedef std::map<std::string, window> window_map;

    _Burst(const std::string& config);
    ~_Burst() {}
    bool add_document(const std::string& str, double pos);
    window get_result(const std::string& keyword) const;
    window get_result_at(const std::string& keyword, double pos) const;
    window_map get_all_bursted_results() const;
    window_map get_all_bursted_results_at(double pos) const;
    keyword_list get_all_keywords() const;
    bool add_keyword(const std::string& keyword, const keyword_params& params);
    bool remove_keyword(const std::string& keyword);
    bool remove_all_keywords();
    void calculate_results();
};

class _Bandit : public _Base<jubatus::core::driver::bandit> {
public:
    _Bandit(const std::string& config);
    ~_Bandit() {}
    bool register_arm(const std::string& arm_id);
    bool delete_arm(const std::string& arm_id);
    std::string select_arm(const std::string& player_id);
    bool register_reward(const std::string& player_id, const std::string& arm_id, double reward);
    std::map<std::string, jubatus::core::bandit::arm_info> get_arm_info(const std::string& player_id) const;
    bool reset(const std::string& player_id);
};
