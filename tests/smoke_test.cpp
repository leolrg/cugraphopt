#include "cugraphopt/core.hpp"
#include "cugraphopt/version.hpp"

#include <cassert>
#include <string>

int main() {
  assert(cugraphopt::version_string() == std::string("0.1.0"));
  assert(cugraphopt::cuda_compilation_unit_marker() == 7);
  const std::string banner = cugraphopt::build_banner();
  assert(banner.find("CuGraphOpt") != std::string::npos);
  assert(banner.find("CUDA marker=7") != std::string::npos);
  return 0;
}
