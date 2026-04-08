# CuGraphOpt Pose Graph Parser Design

Date: 2026-04-08

## Goal

Add the first real data-ingest path to CuGraphOpt by parsing `.g2o` pose-graph files into in-memory CPU-side structures. This parser is intentionally narrow: it only supports the record types needed immediately for 3D pose graph optimization datasets.

## Scope

This design covers:

- CPU-side pose graph data structures
- Parsing `VERTEX_SE3:QUAT` records into node structs
- Parsing `EDGE_SE3:QUAT` records into edge structs
- A parser API that loads a `.g2o` file into memory
- A simple executable path that loads a file and prints graph counts
- Automated tests using a tiny hand-written `.g2o` fixture

This design does not cover:

- GTSAM integration
- Graph validation beyond basic parse correctness
- Sparse matrix assembly
- Linearization or residual evaluation
- GPU upload formats or device-side graph storage
- Support for additional `.g2o` record types

## Recommended Approach

Use a small domain-specific parser inside `cugraphopt_core`.

The parser should:

- read the file line-by-line
- ignore blank lines and comment lines
- parse only `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT`
- treat any other non-empty record type as an error
- return a `PoseGraph` object containing `nodes` and `edges`

This is preferred over a general-purpose `.g2o` parser framework because the current milestone needs a correctness baseline quickly, not an extensible plugin system. It is also preferred over parsing directly into solver-oriented packed arrays because that would prematurely couple file ingestion to future GPU layout decisions.

## Data Model

The parser should introduce three plain data structures:

- `Pose3Node`
- `Pose3Edge`
- `PoseGraph`

`Pose3Node` should contain:

- integer node ID
- translation as three scalars: `x`, `y`, `z`
- orientation quaternion as four scalars: `qx`, `qy`, `qz`, `qw`

`Pose3Edge` should contain:

- integer source node ID
- integer target node ID
- relative translation as `x`, `y`, `z`
- relative quaternion as `qx`, `qy`, `qz`, `qw`
- the upper-triangular information matrix entries exactly as stored in `.g2o` for `EDGE_SE3:QUAT`

`PoseGraph` should contain:

- `std::vector<Pose3Node> nodes`
- `std::vector<Pose3Edge> edges`

The parser should preserve file order rather than reindexing or canonicalizing anything. This keeps the first ingest path simple and faithful to the source dataset.

## Parser API

Expose a small header-level API from `cugraphopt_core`:

- one function to load a pose graph from a filesystem path

The API should throw a descriptive exception on parse failures such as:

- file cannot be opened
- truncated line
- unsupported record type
- numeric extraction failure

For this milestone, exceptions are preferable to status-code plumbing because the main need is fast correctness feedback during development and testing.

## Parsing Behavior

The parser should use line-oriented input via `std::ifstream` and `std::istringstream`.

Behavior rules:

- skip blank lines
- skip comment lines beginning with `#`
- if the first token is `VERTEX_SE3:QUAT`, parse one node
- if the first token is `EDGE_SE3:QUAT`, parse one edge
- otherwise fail immediately with an error naming the unsupported record type

The `EDGE_SE3:QUAT` parser should read exactly the standard payload for a 3D pose-graph edge, including the 21 upper-triangular information matrix values.

The design intentionally does not require:

- duplicate ID detection
- contiguous ID checks
- symmetry expansion of the information matrix
- quaternion normalization

Those checks may be added later if they become necessary for solver correctness.

## Executable Behavior

Extend the existing executable so it can optionally accept a `.g2o` file path.

If no argument is provided, it may keep its current banner behavior or print a short usage message.

If a path is provided, it should:

1. parse the file into a `PoseGraph`
2. print node and edge counts

This CLI is only a smoke path for parser verification. It is not intended to become a full command-line interface.

## Testing Strategy

Add parser-focused automated tests with a tiny fixture file stored under the test tree.

The tests should verify:

- a small valid file parses successfully
- parsed node count matches expectation
- parsed edge count matches expectation
- at least one node field is parsed correctly
- at least one edge field is parsed correctly
- an unsupported record type fails

The fixture should be small enough to inspect manually and should contain:

- at least two `VERTEX_SE3:QUAT` lines
- at least one `EDGE_SE3:QUAT` line

## File Layout

Planned additions:

- public parser/data-model header under `include/cugraphopt/`
- parser implementation file under `src/`
- test fixture under `tests/fixtures/`
- parser test source under `tests/`

The existing core library target should absorb the parser source file. No new library target is needed.

## Risks and Mitigations

### Over-designing the parser

Risk:
Building a flexible `.g2o` framework now will slow down delivery and obscure correctness.

Mitigation:
Support only the two required record types and fail on anything else.

### Hiding malformed input

Risk:
Permissive parsing can make later solver bugs harder to diagnose.

Mitigation:
Treat unsupported records and tokenization failures as immediate parse errors.

### Premature solver coupling

Risk:
If parsing writes directly into future GPU-oriented layouts, future changes become harder.

Mitigation:
Keep the first representation as plain CPU-side structs with vectors.

## Next Step After This Slice

Once the parser is in place and tested, the next milestone should be building the first graph-aware executable path and then adding CPU-side linearization primitives on top of the parsed `PoseGraph`.
