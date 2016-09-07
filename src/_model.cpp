#include <ctime>
#include <jubatus/config.hpp>
#include <jubatus/core/common/big_endian.hpp>
#include <jubatus/server/common/crc32.hpp>

#include "_wrapper.h"

static const size_t MODEL_HEADER_SIZE = 48;
static const char MAGIC_NUMBER[8] = "jubatus";
static const uint64_t MODEL_FORMAT_VERSION = 1;
static const uint64_t MODEL_SYSTEM_VERSION = 1;

std::string pack_model(const std::string& type,
                       const std::string& config,
                       const std::string& id,
                       const msgpack::sbuffer& user_data_buf) {
    // outputs jubatus(not jubatus-core) compatible model
    // ref: https://github.com/jubatus/jubatus/blob/master/jubatus/server/framework/save_load.cpp
    using jubatus::core::framework::stream_writer;
    using jubatus::core::framework::jubatus_packer;
    using jubatus::core::framework::packer;
    using jubatus::core::common::write_big_endian;
    using jubatus::server::common::calc_crc32;

    msgpack::sbuffer system_data_buf;
    {
        stream_writer<msgpack::sbuffer> st(system_data_buf);
        jubatus_packer jp(st);
        packer packer(jp);
        packer.pack_array(5);
        packer.pack_uint64(MODEL_SYSTEM_VERSION);
        packer.pack_uint64(std::time(NULL));
        packer.pack(type);
        packer.pack(id);
        packer.pack(config);
    }

    size_t model_size = MODEL_HEADER_SIZE + system_data_buf.size() + user_data_buf.size();
    std::string model;
    model.resize(model_size);
    char *p = const_cast<char*>(model.data());

    // build header
    {
        uint32_t major, minor, maintenance;
        std::sscanf(JUBATUS_VERSION, "%d.%d.%d", &major, &minor, &maintenance);
        std::memcpy(p, MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
        write_big_endian(MODEL_FORMAT_VERSION, &p[8]);
        write_big_endian(major, &p[16]);
        write_big_endian(minor, &p[20]);
        write_big_endian(maintenance, &p[24]);
        // write_big_endian(crc32, &p[28]);  // skipped
        write_big_endian(static_cast<uint64_t>(system_data_buf.size()), &p[32]);
        write_big_endian(static_cast<uint64_t>(user_data_buf.size()), &p[40]);

        uint32_t crc32 = calc_crc32(p, 28);
        crc32 = calc_crc32(&p[32], 16, crc32);
        crc32 = calc_crc32(system_data_buf.data(), system_data_buf.size(), crc32);
        crc32 = calc_crc32(user_data_buf.data(), user_data_buf.size(), crc32);
        write_big_endian(crc32, &p[28]);
    }

    p += MODEL_HEADER_SIZE;
    std::memcpy(p, system_data_buf.data(), system_data_buf.size());
    p += system_data_buf.size();
    std::memcpy(p, user_data_buf.data(), user_data_buf.size());
    return model;
}

void unpack_model(const std::string& data,
                  msgpack::unpacked& user_data_buffer,
                  std::string& model_type,
                  std::string& model_id,
                  std::string& model_config,
                  uint64_t *user_data_version,
                  msgpack::object **user_data) {
    using jubatus::core::common::read_big_endian;
    using jubatus::server::common::calc_crc32;

    uint32_t major, minor, maintenance;
    std::sscanf(JUBATUS_VERSION, "%d.%d.%d", &major, &minor, &maintenance);

    const char *p = data.data();
    do {
        if (std::memcmp(p, MAGIC_NUMBER, sizeof(MAGIC_NUMBER)) ||
            read_big_endian<uint64_t>(&p[8]) != MODEL_FORMAT_VERSION ||
            read_big_endian<uint32_t>(&p[16]) != major ||
            read_big_endian<uint32_t>(&p[20]) != minor ||
            read_big_endian<uint32_t>(&p[24]) != maintenance) break;
        uint32_t crc32_expected = read_big_endian<uint32_t>(&p[28]);
        uint64_t system_data_size = read_big_endian<uint64_t>(&p[32]);
        uint64_t user_data_size = read_big_endian<uint64_t>(&p[40]);
        if (MODEL_HEADER_SIZE + system_data_size + user_data_size != data.size())
            break;

        const char *sys = &p[MODEL_HEADER_SIZE];
        const char *user = &p[MODEL_HEADER_SIZE + system_data_size];
        uint32_t crc32_actual = calc_crc32(p, 28);
        crc32_actual = calc_crc32(&p[32], 16, crc32_actual);
        crc32_actual = calc_crc32(sys, system_data_size, crc32_actual);
        crc32_actual = calc_crc32(user, user_data_size, crc32_actual);
        if (crc32_actual != crc32_expected)
            break;

        {
            msgpack::unpacked unpacked;
            msgpack::unpack(&unpacked, sys, system_data_size);
            if (unpacked.get().type != msgpack::type::ARRAY)
                break;
            msgpack::object_array& sc = unpacked.get().via.array;
            if (sc.size != 5 || sc.ptr[0].via.u64 != MODEL_SYSTEM_VERSION
                || sc.ptr[2].type != msgpack::type::RAW
                || sc.ptr[3].type != msgpack::type::RAW
                || sc.ptr[4].type != msgpack::type::RAW) break;
            model_type.assign(sc.ptr[2].via.raw.ptr, sc.ptr[2].via.raw.size);
            model_id.assign(sc.ptr[3].via.raw.ptr, sc.ptr[3].via.raw.size);
            model_config.assign(sc.ptr[4].via.raw.ptr, sc.ptr[4].via.raw.size);
        }

        {
            msgpack::unpack(&user_data_buffer, user, user_data_size);
            if (user_data_buffer.get().type != msgpack::type::ARRAY)
                break;
            msgpack::object_array& sc = user_data_buffer.get().via.array;
            if (sc.size != 2 || sc.ptr[0].type != msgpack::type::POSITIVE_INTEGER)
                break;
            *user_data_version = sc.ptr[0].via.u64;
            *user_data = &sc.ptr[1];
        }
        return;
    } while (false);
    throw std::runtime_error("invalid format");
}
