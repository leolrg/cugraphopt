#pragma once

#include <string>
#include <string_view>

namespace cugraphopt {

inline constexpr std::string_view kProjectName = "CuGraphOpt";
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

std::string version_string();

}  // namespace cugraphopt
