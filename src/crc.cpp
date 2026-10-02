#include "crc.hpp"
#include "cppcrc.h"


namespace ww {
namespace crc {

    uint16_t Encode(const char* data, size_t sz) {
        return CRC16::CCITT_FALSE::calc(reinterpret_cast<const uint8_t*>(data), sz);
    }

    bool Verify(const char* data, size_t sz, uint16_t crc_code) {
        return Encode(data, sz) == crc_code;
    }

} // namespace crc
} // namespace ww
