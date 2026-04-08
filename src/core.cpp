#include "cugraphopt/core.hpp"
#include "cugraphopt/version.hpp"

#include <string>

namespace cugraphopt {

std::string version_string() {
  return std::to_string(kVersionMajor) + "." + std::to_string(kVersionMinor) +
         "." + std::to_string(kVersionPatch);
}

std::string build_banner() {
  return std::string(kProjectName) + " v" + version_string() +
         " (CUDA marker=" + std::to_string(cuda_compilation_unit_marker()) + ")";
}

}  // namespace cugraphopt
