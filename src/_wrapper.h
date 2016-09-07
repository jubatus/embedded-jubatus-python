#include <string>

#include <jubatus/util/lang/shared_ptr.h>
#include <jubatus/util/text/json.h>
#include <jubatus/core/fv_converter/datum.hpp>
#include <jubatus/core/framework/stream_writer.hpp>
#include <jubatus/core/driver/classifier.hpp>

using jubatus::util::lang::shared_ptr;
using jubatus::core::fv_converter::datum;
using jubatus::core::classifier::classify_result;

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

    void load(const std::string& data, const std::string& type, uint64_t version) {
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
