#include <sys/types.h>
#include <unistd.h>

#include <string>

#include <jubatus/util/lang/shared_ptr.h>
#include <jubatus/util/system/time_util.h>
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
#include <jubatus/core/driver/stat.hpp>
#include <jubatus/core/driver/weight.hpp>
#include <jubatus/core/driver/graph.hpp>

using jubatus::util::lang::shared_ptr;
using jubatus::util::system::time::clock_time;
using jubatus::util::system::time::get_clock_time;
using jubatus::core::common::sfv_t;
using jubatus::core::fv_converter::datum;
using jubatus::core::classifier::classify_result;
using jubatus::core::clustering::cluster_unit;
using jubatus::core::clustering::cluster_set;
using jubatus::core::clustering::index_cluster_set;
using jubatus::core::clustering::index_cluster_unit;
using jubatus::core::clustering::indexed_point;
using jubatus::core::graph::node_id_t;
using jubatus::core::graph::edge_id_t;
typedef jubatus::core::graph::property property_t;

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
    virtual void get_status_(std::map<std::string, std::string>& status) const = 0;
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

    std::map<std::string, std::map<std::string, std::string> > get_status() {
        std::map<std::string, std::map<std::string, std::string> > status;
        std::map<std::string, std::string>& data = status["embedded"];

        const clock_time ct = get_clock_time();
        data["clock_time"] =
            jubatus::util::lang::lexical_cast<std::string>(ct.sec);
        data["pid"] = jubatus::util::lang::lexical_cast<std::string>(getpid());

        this->get_status_(data);

        return status;
    }
};

typedef std::vector<std::pair<std::string, double> > id_score_list_t;

class _Classifier : public _Base<jubatus::core::driver::classifier> {
public:
    _Classifier(const std::string& config);
    ~_Classifier() {}
    void train(const std::string& label, const datum& d);
    classify_result classify(const datum& d);
    std::vector<std::pair<std::string, uint64_t> > get_labels();
    bool set_label(const std::string& new_label);
    bool delete_label(const std::string& target_label);
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Regression : public _Base<jubatus::core::driver::regression> {
public:
    _Regression(const std::string& config);
    ~_Regression() {}
    void train(double score, const datum& d);
    double estimate(const datum& d);
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
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
    id_score_list_t similar_row_from_id_and_score(const std::string& id, double score);
    id_score_list_t similar_row_from_id_and_rate(const std::string& id, float rate);
    id_score_list_t similar_row_from_datum(const datum& d, size_t ret_num);
    id_score_list_t similar_row_from_datum_and_score(const datum& d, double score);
    id_score_list_t similar_row_from_datum_and_rate(const datum& d, float rate);
    datum decode_row(const std::string& id);
    std::vector<std::string> get_all_rows();
    double calc_similarity(const datum& l, const datum& r);
    double calc_l2norm(const datum& d);
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
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
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Anomaly : public _Base<jubatus::core::driver::anomaly> {
    uint64_t idgen;
public:
    _Anomaly(const std::string& config);
    ~_Anomaly() {}
    void load(const std::string& data, const std::string& type, uint64_t version);

    void clear_row(const std::string& id);
    std::pair<std::string, double> add(const datum& d);
    std::vector<std::string> add_bulk(const std::vector<std::pair<std::string, datum> >& data);
    double update(const std::string& id, const datum& d);
    double overwrite(const std::string &id, const datum& d);
    double calc_score(const datum& d) const;
    std::vector<std::string> get_all_rows() const;
    inline std::string get_next_id() { return jubatus::util::lang::lexical_cast<std::string>(idgen++); }
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Clustering : public _Base<jubatus::core::driver::clustering> {
public:
    _Clustering(const std::string& config);
    ~_Clustering() {}
    void push(const std::vector<indexed_point>& points);
    size_t get_revision() const;
    cluster_set get_core_members() const;
    std::vector<datum> get_k_center() const;
    datum get_nearest_center(const datum& d) const;
    cluster_unit get_nearest_members(const datum& d) const;
    index_cluster_set get_core_members_light() const;
    index_cluster_unit get_nearest_members_light(const datum& d) const;
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
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
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
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
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Stat : public _Base<jubatus::core::driver::stat> {
public:
    _Stat(const std::string& config);
    ~_Stat() {}
    void push(const std::string& key, double value);
    double sum(const std::string& key) const;
    double stddev(const std::string& key) const;
    double max(const std::string& key) const;
    double min(const std::string& key) const;
    double entropy() const;
    double moment(const std::string& key, int degree, double center) const;
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Weight : public _Base<jubatus::core::driver::weight> {
public:
    _Weight(const std::string& config);
    ~_Weight() {}
    sfv_t update(const datum&);
    sfv_t calc_weight(const datum&) const;
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};

class _Graph : public _Base<jubatus::core::driver::graph> {
    uint64_t id_;

    inline uint64_t generate_id() { return id_++; }
public:
    _Graph(const std::string& config);
    ~_Graph() {}
    void load(const std::string& data, const std::string& type, uint64_t version);

    std::string create_node();
    bool remove_node(const std::string& node_id);
    bool update_node(const std::string& node_id,
                     const property_t& property);
    edge_id_t create_edge(const std::string& src,
                          const std::string& target,
                          const property_t& property);
    bool update_edge(edge_id_t edge_id,
                     const property_t& property);
    void remove_edge(edge_id_t edge_id);

    double get_centrality(const std::string& node_id,
                          int centrality_type,
                          const jubatus::core::graph::preset_query& q) const;
    void add_centrality_query(const jubatus::core::graph::preset_query& q);
    void add_shortest_path_query(const jubatus::core::graph::preset_query& q);
    void remove_centrality_query(const jubatus::core::graph::preset_query& q);
    void remove_shortest_path_query(const jubatus::core::graph::preset_query& q);
    std::vector<node_id_t> get_shortest_path(const std::string& src,
                                             const std::string& target,
                                             uint64_t max_hop,
                                             const jubatus::core::graph::preset_query &q) const;

    void update_index();
    jubatus::core::graph::node_info get_node(const std::string& node_id) const;
    jubatus::core::graph::edge_info get_edge(edge_id_t eid) const;
protected:
    void get_status_(std::map<std::string, std::string>& status) const;
};
