#include "cugraphopt/pose_graph.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cugraphopt {
namespace {

std::string_view trim(std::string_view value) {
  std::size_t start = 0;
  while (start < value.size() &&
         std::isspace(static_cast<unsigned char>(value[start])) != 0) {
    ++start;
  }

  std::size_t end = value.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }

  return value.substr(start, end - start);
}

[[noreturn]] void throw_parse_error(const std::size_t line_number,
                                    const std::string& message) {
  throw std::runtime_error("Parse error on line " + std::to_string(line_number) +
                           ": " + message);
}

template <typename T>
void read_value(std::istringstream& stream, T& value,
                const std::size_t line_number, const std::string& field_name) {
  if (!(stream >> value)) {
    throw_parse_error(line_number, "failed to read " + field_name);
  }
}

void ensure_no_extra_tokens(std::istringstream& stream,
                            const std::size_t line_number) {
  std::string extra_token;
  if (stream >> extra_token) {
    throw_parse_error(line_number,
                      "unexpected trailing token '" + extra_token + "'");
  }
}

Pose3Node parse_vertex(std::istringstream& stream,
                       const std::size_t line_number) {
  Pose3Node node;
  read_value(stream, node.id, line_number, "vertex id");
  read_value(stream, node.x, line_number, "vertex x");
  read_value(stream, node.y, line_number, "vertex y");
  read_value(stream, node.z, line_number, "vertex z");
  read_value(stream, node.qx, line_number, "vertex qx");
  read_value(stream, node.qy, line_number, "vertex qy");
  read_value(stream, node.qz, line_number, "vertex qz");
  read_value(stream, node.qw, line_number, "vertex qw");
  ensure_no_extra_tokens(stream, line_number);
  return node;
}

Pose3Edge parse_edge(std::istringstream& stream, const std::size_t line_number) {
  Pose3Edge edge;
  read_value(stream, edge.from, line_number, "edge from");
  read_value(stream, edge.to, line_number, "edge to");
  read_value(stream, edge.x, line_number, "edge x");
  read_value(stream, edge.y, line_number, "edge y");
  read_value(stream, edge.z, line_number, "edge z");
  read_value(stream, edge.qx, line_number, "edge qx");
  read_value(stream, edge.qy, line_number, "edge qy");
  read_value(stream, edge.qz, line_number, "edge qz");
  read_value(stream, edge.qw, line_number, "edge qw");

  for (std::size_t index = 0; index < edge.information.size(); ++index) {
    read_value(stream, edge.information[index], line_number,
               "edge information[" + std::to_string(index) + "]");
  }

  ensure_no_extra_tokens(stream, line_number);
  return edge;
}

}  // namespace

PoseGraph load_pose_graph(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open pose graph file: " + path.string());
  }

  PoseGraph graph;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::string_view trimmed = trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    std::istringstream stream{std::string(trimmed)};
    std::string record_type;
    read_value(stream, record_type, line_number, "record type");

    if (record_type == "VERTEX_SE3:QUAT") {
      graph.nodes.push_back(parse_vertex(stream, line_number));
      continue;
    }

    if (record_type == "EDGE_SE3:QUAT") {
      graph.edges.push_back(parse_edge(stream, line_number));
      continue;
    }

    throw_parse_error(line_number, "unsupported record type '" + record_type +
                                       "'");
  }

  return graph;
}

}  // namespace cugraphopt
