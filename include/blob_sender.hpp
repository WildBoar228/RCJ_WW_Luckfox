#ifndef _RCJ_WW_LUCKFOX_INCLUDE_BLOB_SENDER_HPP_
#define _RCJ_WW_LUCKFOX_INCLUDE_BLOB_SENDER_HPP_

#include <iostream>
#include <vector>

#include "rcj_vision.hpp"

namespace ww {
namespace vision {

    void SendBlobs(std::ostream&, const std::vector<std::vector<BlobGeom>>&);
    void Send2BlobInfo(std::ostream& out,  BlobInfo b1,  BlobInfo b2);

} // namespace vision
} // namespace ww

#endif
