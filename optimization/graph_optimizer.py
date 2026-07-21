#!/usr/bin/env python3
"""
Inference Graph Optimizer for Local LLMs.

Features:
- Operator fusion
- Constant folding
- Dead code elimination
- Memory layout optimization
- Execution scheduling

Usage:
    from optimization.graph_optimizer import GraphOptimizer

    optimizer = GraphOptimizer()
    optimized_graph = optimizer.optimize(model_graph)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraphNode:
    """Graph node for computation."""
    id: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ComputationGraph:
    """Computation graph."""
    nodes: list[GraphNode] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


class GraphOptimizer:
    """
    Optimizer for inference computation graphs.

    Features:
    - Operator fusion
    - Constant folding
    - Memory optimization
    - Parallel execution planning
    """

    def __init__(self):
        self.optimization_stats = {
            "nodes_before": 0,
            "nodes_after": 0,
            "fusions": 0,
            "constants_folded": 0,
        }

    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Apply all optimizations to graph.

        Args:
            graph: Input computation graph

        Returns:
            Optimized graph
        """
        self.optimization_stats["nodes_before"] = len(graph.nodes)

        # Apply optimizations
        graph = self.fuse_operators(graph)
        graph = self.fold_constants(graph)
        graph = self.eliminate_dead_code(graph)
        graph = self.optimize_memory_layout(graph)

        self.optimization_stats["nodes_after"] = len(graph.nodes)

        return graph

    def fuse_operators(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Fuse compatible operators.

        Args:
            graph: Input graph

        Returns:
            Fused graph
        """
        fused_nodes = []
        i = 0

        while i < len(graph.nodes):
            node = graph.nodes[i]

            # Check for fusible patterns
            if i + 1 < len(graph.nodes):
                next_node = graph.nodes[i + 1]

                # Fuse MatMul + Add -> Linear
                if node.op_type == "MatMul" and next_node.op_type == "Add":
                    if node.outputs[0] in next_node.inputs:
                        fused = GraphNode(
                            id=f"fused_{node.id}_{next_node.id}",
                            op_type="Linear",
                            inputs=node.inputs + next_node.inputs[1:],
                            outputs=next_node.outputs,
                            attributes={"fused": True},
                        )
                        fused_nodes.append(fused)
                        self.optimization_stats["fusions"] += 1
                        i += 2
                        continue

                # Fuse LayerNorm + Dropout
                if node.op_type == "LayerNorm" and next_node.op_type == "Dropout":
                    if node.outputs[0] in next_node.inputs:
                        fused = GraphNode(
                            id=f"fused_{node.id}_{next_node.id}",
                            op_type="LayerNorm",
                            inputs=node.inputs,
                            outputs=next_node.outputs,
                            attributes={"dropout": 0.0},
                        )
                        fused_nodes.append(fused)
                        self.optimization_stats["fusions"] += 1
                        i += 2
                        continue

            fused_nodes.append(node)
            i += 1

        return ComputationGraph(
            nodes=fused_nodes,
            inputs=graph.inputs,
            outputs=graph.outputs,
        )

    def fold_constants(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Fold constant operations.

        Args:
            graph: Input graph

        Returns:
            Optimized graph
        """
        folded_nodes = []
        constants = {}

        # First pass: identify constants
        for node in graph.nodes:
            if node.op_type == "Constant":
                constants[node.outputs[0]] = node.attributes.get("value")

        # Second pass: fold operations on constants
        for node in graph.nodes:
            if node.op_type in ["Add", "Mul", "Sub", "Div"]:
                # Check if all inputs are constants
                all_constant = all(inp in constants for inp in node.inputs)
                if all_constant:
                    # Compute result
                    result = self._compute_constant_op(
                        node.op_type,
                        [constants[inp] for inp in node.inputs],
                    )
                    constants[node.outputs[0]] = result
                    self.optimization_stats["constants_folded"] += 1
                    continue

            folded_nodes.append(node)

        return ComputationGraph(
            nodes=folded_nodes,
            inputs=graph.inputs,
            outputs=graph.outputs,
        )

    def _compute_constant_op(self, op_type: str, operands: list[Any]) -> Any:
        """Compute constant operation."""
        if op_type == "Add":
            return sum(operands)
        elif op_type == "Mul":
            result = operands[0]
            for op in operands[1:]:
                result *= op
            return result
        elif op_type == "Sub":
            return operands[0] - sum(operands[1:])
        elif op_type == "Div":
            result = operands[0]
            for op in operands[1:]:
                result /= op
            return result
        return None

    def eliminate_dead_code(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Eliminate unused nodes.

        Args:
            graph: Input graph

        Returns:
            Optimized graph
        """
        # Find used outputs
        used_outputs = set(graph.outputs)
        for node in graph.nodes:
            for inp in node.inputs:
                used_outputs.add(inp)

        # Keep only nodes that produce used outputs
        alive_nodes = []
        for node in graph.nodes:
            if any(out in used_outputs for out in node.outputs):
                alive_nodes.append(node)

        return ComputationGraph(
            nodes=alive_nodes,
            inputs=graph.inputs,
            outputs=graph.outputs,
        )

    def optimize_memory_layout(self, graph: ComputationGraph) -> ComputationGraph:
        """
        Optimize memory layout for better cache performance.

        Args:
            graph: Input graph

        Returns:
            Optimized graph
        """
        # Reorder nodes for better memory locality
        # Simple heuristic: group by operation type
        op_groups = {}
        for node in graph.nodes:
            if node.op_type not in op_groups:
                op_groups[node.op_type] = []
            op_groups[node.op_type].append(node)

        # Rebuild graph with grouped operations
        optimized_nodes = []
        for op_type, nodes in op_groups.items():
            optimized_nodes.extend(nodes)

        return ComputationGraph(
            nodes=optimized_nodes,
            inputs=graph.inputs,
            outputs=graph.outputs,
        )

    def get_execution_order(self, graph: ComputationGraph) -> list[str]:
        """
        Determine optimal execution order.

        Args:
            graph: Computation graph

        Returns:
            List of node IDs in execution order
        """
        # Topological sort
        in_degree = {node.id: 0 for node in graph.nodes}
        node_map = {node.id: node for node in graph.nodes}

        for node in graph.nodes:
            for inp in node.inputs:
                if inp in in_degree:
                    in_degree[node.id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)

            node = node_map.get(nid)
            if node:
                for out in node.outputs:
                    for other_node in graph.nodes:
                        if out in other_node.inputs:
                            in_degree[other_node.id] -= 1
                            if in_degree[other_node.id] == 0:
                                queue.append(other_node.id)

        return order

    def get_optimization_report(self) -> dict:
        """Get optimization statistics."""
        return {
            **self.optimization_stats,
            "reduction": (
                (self.optimization_stats["nodes_before"] - self.optimization_stats["nodes_after"])
                / max(1, self.optimization_stats["nodes_before"])
                * 100
            ),
        }


class ModelGraphBuilder:
    """Build computation graphs from model structure."""

    @staticmethod
    def build_transformer_layer(
        layer_id: int,
        hidden_dim: int,
        num_heads: int,
    ) -> ComputationGraph:
        """Build graph for transformer layer."""
        nodes = [
            GraphNode(
                id=f"layer{layer_id}_norm1",
                op_type="LayerNorm",
                inputs=[f"input_{layer_id}"],
                outputs=[f"norm1_{layer_id}"],
            ),
            GraphNode(
                id=f"layer{layer_id}_attn",
                op_type="MultiHeadAttention",
                inputs=[f"norm1_{layer_id}"],
                outputs=[f"attn_{layer_id}"],
                attributes={"num_heads": num_heads},
            ),
            GraphNode(
                id=f"layer{layer_id}_add1",
                op_type="Add",
                inputs=[f"input_{layer_id}", f"attn_{layer_id}"],
                outputs=[f"residual1_{layer_id}"],
            ),
            GraphNode(
                id=f"layer{layer_id}_norm2",
                op_type="LayerNorm",
                inputs=[f"residual1_{layer_id}"],
                outputs=[f"norm2_{layer_id}"],
            ),
            GraphNode(
                id=f"layer{layer_id}_ffn",
                op_type="FeedForward",
                inputs=[f"norm2_{layer_id}"],
                outputs=[f"ffn_{layer_id}"],
                attributes={"hidden_dim": hidden_dim * 4},
            ),
            GraphNode(
                id=f"layer{layer_id}_add2",
                op_type="Add",
                inputs=[f"residual1_{layer_id}", f"ffn_{layer_id}"],
                outputs=[f"output_{layer_id}"],
            ),
        ]

        return ComputationGraph(
            nodes=nodes,
            inputs=[f"input_{layer_id}"],
            outputs=[f"output_{layer_id}"],
        )

    @staticmethod
    def build_full_model(num_layers: int = 12, hidden_dim: int = 768, num_heads: int = 12) -> ComputationGraph:
        """Build full model graph."""
        all_nodes = []
        prev_output = "input"

        for i in range(num_layers):
            layer_graph = ModelGraphBuilder.build_transformer_layer(i, hidden_dim, num_heads)
            # Update input/output connections
            for node in layer_graph.nodes:
                node.inputs = [prev_output if inp == f"input_{i}" else inp for inp in node.inputs]
                all_nodes.append(node)
            prev_output = f"output_{i}"

        return ComputationGraph(
            nodes=all_nodes,
            inputs=["input"],
            outputs=[prev_output],
        )


def main(argv=None):
    """Graph optimizer command-line interface."""
    import argparse
    import json
    import os
    import sys
    from dataclasses import asdict

    def resolve(s):
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            if os.path.exists(s):
                with open(s) as f:
                    return json.load(f)
            raise

    def build_graph(d):
        nodes = [
            GraphNode(
                id=n["id"],
                op_type=n["op_type"],
                inputs=list(n.get("inputs", [])),
                outputs=list(n.get("outputs", [])),
                attributes=dict(n.get("attributes", {})),
                metadata=dict(n.get("metadata", {})),
            )
            for n in d.get("nodes", [])
        ]
        return ComputationGraph(
            nodes=nodes,
            inputs=list(d.get("inputs", [])),
            outputs=list(d.get("outputs", [])),
        )

    parser = argparse.ArgumentParser(description="Inference graph optimizer for local LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    op = sub.add_parser("optimize")
    op.add_argument("--graph", required=True, help="Computation graph (JSON or file path)")

    od = sub.add_parser("order")
    od.add_argument("--graph", required=True, help="Computation graph (JSON or file path)")

    bd = sub.add_parser("build")
    bd.add_argument("--num-layers", type=int, default=12)
    bd.add_argument("--hidden-dim", type=int, default=768)
    bd.add_argument("--num-heads", type=int, default=12)

    args = parser.parse_args(argv)

    try:
        if args.command == "optimize":
            graph = build_graph(resolve(args.graph))
            optimizer = GraphOptimizer()
            optimized = optimizer.optimize(graph)
            result = {
                "nodes": [asdict(n) for n in optimized.nodes],
                "inputs": optimized.inputs,
                "outputs": optimized.outputs,
                "report": optimizer.get_optimization_report(),
            }
        elif args.command == "order":
            graph = build_graph(resolve(args.graph))
            result = GraphOptimizer().get_execution_order(graph)
        else:
            graph = ModelGraphBuilder.build_full_model(
                num_layers=args.num_layers, hidden_dim=args.hidden_dim, num_heads=args.num_heads
            )
            result = {
                "nodes": [asdict(n) for n in graph.nodes],
                "inputs": graph.inputs,
                "outputs": graph.outputs,
            }
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
