#!/usr/bin/env python3
"""Generate synthetic g2o pose graph datasets for benchmarking."""

import argparse
import math
import random
import sys

def gen_sphere(n_nodes, noise_sigma=0.05, loop_ratio=0.5):
    """Generate a sphere-like topology: chain + random loop closures."""
    random.seed(42)
    nodes = []
    edges = []

    # Place nodes on a 3D spiral
    for i in range(n_nodes):
        t = i / n_nodes * 4 * math.pi
        r = 10.0
        x = r * math.cos(t) + random.gauss(0, noise_sigma)
        y = r * math.sin(t) + random.gauss(0, noise_sigma)
        z = i * 0.1 + random.gauss(0, noise_sigma)
        nodes.append((i, x, y, z, 0.0, 0.0, 0.0, 1.0))

    # Odometry edges (chain)
    for i in range(n_nodes - 1):
        dx = nodes[i+1][1] - nodes[i][1]
        dy = nodes[i+1][2] - nodes[i][2]
        dz = nodes[i+1][3] - nodes[i][3]
        edges.append((i, i+1, dx, dy, dz, 0.0, 0.0, 0.0, 1.0))

    # Loop closure edges
    n_loops = int(n_nodes * loop_ratio)
    for _ in range(n_loops):
        i = random.randint(0, n_nodes - 1)
        j = random.randint(0, n_nodes - 1)
        if abs(i - j) < 5:
            continue
        if i > j:
            i, j = j, i
        dx = nodes[j][1] - nodes[i][1] + random.gauss(0, noise_sigma)
        dy = nodes[j][2] - nodes[i][2] + random.gauss(0, noise_sigma)
        dz = nodes[j][3] - nodes[i][3] + random.gauss(0, noise_sigma)
        edges.append((i, j, dx, dy, dz, 0.0, 0.0, 0.0, 1.0))

    return nodes, edges

def write_g2o(filename, nodes, edges):
    """Write nodes and edges in g2o format."""
    # Identity information matrix (upper triangular, 21 values)
    info = " ".join(["500" if i == j else "0"
                     for i in range(6) for j in range(i, 6)])

    with open(filename, 'w') as f:
        for node in nodes:
            f.write(f"VERTEX_SE3:QUAT {node[0]} {node[1]:.6f} {node[2]:.6f} "
                    f"{node[3]:.6f} {node[4]:.6f} {node[5]:.6f} "
                    f"{node[6]:.6f} {node[7]:.6f}\n")
        for edge in edges:
            f.write(f"EDGE_SE3:QUAT {edge[0]} {edge[1]} {edge[2]:.6f} "
                    f"{edge[3]:.6f} {edge[4]:.6f} {edge[5]:.6f} "
                    f"{edge[6]:.6f} {edge[7]:.6f} {edge[8]:.6f} {info}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic g2o datasets")
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--output", type=str, default="synthetic.g2o")
    parser.add_argument("--loop-ratio", type=float, default=0.5)
    args = parser.parse_args()

    nodes, edges = gen_sphere(args.nodes, loop_ratio=args.loop_ratio)
    write_g2o(args.output, nodes, edges)
    print(f"Generated {args.output}: {len(nodes)} nodes, {len(edges)} edges")

if __name__ == "__main__":
    main()
