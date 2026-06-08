#ifndef _RCJ_WW_LUCKFOX_INCLUDE_BLOB_SENDER_HPP_
#define _RCJ_WW_LUCKFOX_INCLUDE_BLOB_SENDER_HPP_

#include <iostream>
#include <vector>

#include "rcj_vision.hpp"

namespace ww_vision {

    void SendBlobs(std::ostream&, const std::vector<ww_vision::BlobGeom>&);

} // namespace ww_vision

#endif
