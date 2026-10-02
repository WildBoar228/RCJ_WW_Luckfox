#ifndef _RCJ_WW_LUCKFOX_INCLUDE_CRC_HPP_
#define _RCJ_WW_LUCKFOX_INCLUDE_CRC_HPP_

#include <cstdint>
#include <cstdlib>

namespace ww {
namespace crc {

    uint16_t Encode(const char* data, size_t sz);
    bool Verify(const char* data, size_t sz, uint16_t crc_code);

} // namespace crc
} // namespace ww

#endif
