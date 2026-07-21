"""
Obsidian-like Graph View System for Memory, Data, Skills, and MCPs.

Provides knowledge graph visualization and navigation with:
- Interactive node-edge graph rendering
- Force-directed layout
- Community detection
- Path visualization
- Export to HTML/SVG/JSON
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict
from enum import Enum
import hashlib


class NodeType(Enum):
    MEMORY = "memory"
    DATA = "data"
    SKILL = "skill"
    MCP = "mcp"
    CONCEPT = "concept"
    FILE = "file"
    TOOL = "tool"
    AGENT = "agent"


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: NodeType
    metadata: dict = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    size: float = 10.0
    color: str = "#4a9eff"
    connections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "size": self.size,
            "color": self.color,
            "metadata": self.metadata,
            "tags": self.tags,
            "connections": self.connections,
            "access_count": self.access_count,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "related"
    metadata: dict = field(default_factory=dict)
    directed: bool = False


@dataclass
class Community:
    id: int
    nodes: list[str]
    label: str = ""
    color: str = "#ff6b6b"


class KnowledgeGraph:
    """
    Obsidian-like knowledge graph for memory, data, skills, and MCPs.

    Features:
    - Force-directed graph layout
    - Community detection (Louvain-like)
    - A* pathfinding between nodes
    - Interactive visualization export
    - Real-time updates
    - Graph analytics (centrality, clustering)
    """

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.communities: list[Community] = []
        self._node_counter = 0

    # === Node Management ===

    def add_node(
        self,
        label: str,
        node_type: NodeType,
        metadata: dict = None,
        tags: list[str] = None,
        node_id: str = None,
    ) -> GraphNode:
        """Add a node to the graph."""
        if node_id is None:
            self._node_counter += 1
            node_id = f"node_{self._node_counter}_{hashlib.md5(label.encode()).hexdigest()[:8]}"

        colors = {
            NodeType.MEMORY: "#4a9eff",
            NodeType.DATA: "#7c3aed",
            NodeType.SKILL: "#10b981",
            NodeType.MCP: "#f59e0b",
            NodeType.CONCEPT: "#ef4444",
            NodeType.FILE: "#6b7280",
            NodeType.TOOL: "#ec4899",
            NodeType.AGENT: "#8b5cf6",
        }

        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            metadata=metadata or {},
            tags=tags or [],
            color=colors.get(node_type, "#4a9eff"),
            size=15.0 if node_type in [NodeType.SKILL, NodeType.MCP] else 10.0,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "related",
        directed: bool = False,
    ) -> bool:
        """Add an edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        edge = GraphEdge(
            source=source_id,
            target=target_id,
            weight=weight,
            edge_type=edge_type,
            directed=directed,
        )
        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
        if not directed:
            self.adjacency[target_id].append(source_id)

        # Update node connections
        self.nodes[source_id].connections.append(target_id)
        if not directed:
            self.nodes[target_id].connections.append(source_id)

        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self.nodes:
            return False

        # Remove all edges connected to this node
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
        self.adjacency.pop(node_id, None)
        for nid in list(self.adjacency.keys()):
            self.adjacency[nid] = [n for n in self.adjacency[nid] if n != node_id]

        del self.nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[GraphNode]:
        """Get all neighbors of a node."""
        neighbor_ids = self.adjacency.get(node_id, [])
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    # === Pathfinding (A*, Dijkstra, BFS, DFS) ===

    def find_path_astar(
        self, start_id: str, goal_id: str, heuristic: str = "euclidean"
    ) -> Optional[list[str]]:
        """A* pathfinding algorithm."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        def h(node_id: str) -> float:
            n = self.nodes[node_id]
            g = self.nodes[goal_id]
            if heuristic == "euclidean":
                return math.sqrt((n.x - g.x) ** 2 + (n.y - g.y) ** 2)
            elif heuristic == "manhattan":
                return abs(n.x - g.x) + abs(n.y - g.y)
            elif heuristic == "connections":
                return max(0, len(g.connections) - len(n.connections)) * 0.1
            return 0

        open_set = {start_id}
        came_from: dict[str, str] = {}
        g_score: dict[str, float] = {start_id: 0}
        f_score: dict[str, float] = {start_id: h(start_id)}

        while open_set:
            current = min(open_set, key=lambda n: f_score.get(n, float("inf")))

            if current == goal_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            open_set.remove(current)

            for neighbor in self.adjacency.get(current, []):
                edge_weight = self._get_edge_weight(current, neighbor)
                tentative_g = g_score.get(current, float("inf")) + edge_weight

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h(neighbor)
                    open_set.add(neighbor)

        return None

    def find_path_dijkstra(self, start_id: str, goal_id: str) -> Optional[list[str]]:
        """Dijkstra's shortest path algorithm."""
        import heapq

        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        distances = {start_id: 0}
        came_from: dict[str, str] = {}
        heap = [(0, start_id)]
        visited = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    edge_weight = self._get_edge_weight(current, neighbor)
                    new_dist = dist + edge_weight
                    if new_dist < distances.get(neighbor, float("inf")):
                        distances[neighbor] = new_dist
                        came_from[neighbor] = current
                        heapq.heappush(heap, (new_dist, neighbor))

        return None

    def find_path_bfs(self, start_id: str, goal_id: str) -> Optional[list[str]]:
        """BFS shortest path (unweighted)."""
        from collections import deque

        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current, path = queue.popleft()
            if current == goal_id:
                return path
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def find_path_dfs(self, start_id: str, goal_id: str) -> Optional[list[str]]:
        """DFS path finding."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return None

        stack = [(start_id, [start_id])]
        visited = set()

        while stack:
            current, path = stack.pop()
            if current == goal_id:
                return path
            if current not in visited:
                visited.add(current)
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        return None

    def _get_edge_weight(self, source: str, target: str) -> float:
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge.weight
            if not edge.directed and edge.source == target and edge.target == source:
                return edge.weight
        return 1.0

    # === Force-Directed Layout ===

    def layout_force_directed(
        self,
        iterations: int = 100,
        repulsion: float = 1000.0,
        attraction: float = 0.01,
        damping: float = 0.9,
        center_pull: float = 0.001,
    ):
        """Apply force-directed graph layout."""
        import random

        # Initialize random positions
        for node in self.nodes.values():
            node.x = random.uniform(-100, 100)
            node.y = random.uniform(-100, 100)
            node.vx = 0
            node.vy = 0

        for _ in range(iterations):
            # Repulsion between all nodes
            node_list = list(self.nodes.values())
            for i, n1 in enumerate(node_list):
                for n2 in node_list[i + 1 :]:
                    dx = n2.x - n1.x
                    dy = n2.y - n1.y
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.1)
                    force = repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    n1.vx -= fx
                    n1.vy -= fy
                    n2.vx += fx
                    n2.vy += fy

            # Attraction along edges
            for edge in self.edges:
                if edge.source in self.nodes and edge.target in self.nodes:
                    n1 = self.nodes[edge.source]
                    n2 = self.nodes[edge.target]
                    dx = n2.x - n1.x
                    dy = n2.y - n1.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    force = attraction * dist * edge.weight
                    fx = force * dx / dist if dist > 0 else 0
                    fy = force * dy / dist if dist > 0 else 0
                    n1.vx += fx
                    n1.vy += fy
                    n2.vx -= fx
                    n2.vy -= fy

            # Center pull
            for node in self.nodes.values():
                node.vx -= node.x * center_pull
                node.vy -= node.y * center_pull

            # Apply velocities
            for node in self.nodes.values():
                node.vx *= damping
                node.vy *= damping
                node.x += node.vx
                node.y += node.vy

    # === Community Detection ===

    def detect_communities(self) -> list[Community]:
        """Detect communities using label propagation."""
        if not self.nodes:
            return []

        # Initialize each node as its own community
        community_map = {nid: i for i, nid in enumerate(self.nodes.keys())}

        for _ in range(50):
            changed = False
            for node_id in self.nodes:
                neighbor_communities = defaultdict(int)
                for neighbor in self.adjacency.get(node_id, []):
                    if neighbor in community_map:
                        neighbor_communities[community_map[neighbor]] += 1

                if neighbor_communities:
                    best_community = max(neighbor_communities, key=neighbor_communities.get)
                    if community_map[node_id] != best_community:
                        community_map[node_id] = best_community
                        changed = True

            if not changed:
                break

        # Group nodes by community
        community_groups = defaultdict(list)
        for node_id, community_id in community_map.items():
            community_groups[community_id].append(node_id)

        self.communities = []
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#dfe6e9", "#fd79a8", "#a29bfe"]
        for i, (cid, nodes) in enumerate(community_groups.items()):
            self.communities.append(Community(
                id=i,
                nodes=nodes,
                label=f"Community {i}",
                color=colors[i % len(colors)],
            ))

        return self.communities

    # === Graph Analytics ===

    def calculate_centrality(self) -> dict[str, float]:
        """Calculate betweenness centrality for all nodes."""
        centrality = {nid: 0.0 for nid in self.nodes}

        for source in self.nodes:
            for target in self.nodes:
                if source == target:
                    continue
                path = self.find_path_bfs(source, target)
                if path:
                    for node in path[1:-1]:
                        centrality[node] += 1

        # Normalize
        n = len(self.nodes)
        if n > 2:
            max_centrality = max(centrality.values()) if centrality else 1
            if max_centrality > 0:
                for nid in centrality:
                    centrality[nid] /= max_centrality

        return centrality

    def get_orphan_nodes(self) -> list[GraphNode]:
        """Find nodes with no connections."""
        return [n for n in self.nodes.values() if not self.adjacency.get(n.id, [])]

    def get_hub_nodes(self, min_connections: int = 5) -> list[GraphNode]:
        """Find highly connected nodes."""
        return [n for n in self.nodes.values() if len(self.adjacency.get(n.id, [])) >= min_connections]

    # === Search & Query ===

    def search(self, query: str, node_type: NodeType = None) -> list[GraphNode]:
        """Search nodes by label, tags, or metadata."""
        query_lower = query.lower()
        results = []
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if query_lower in node.label.lower():
                results.append(node)
            elif any(query_lower in tag.lower() for tag in node.tags):
                results.append(node)
            elif any(query_lower in str(v).lower() for v in node.metadata.values()):
                results.append(node)
        return results

    def get_subgraph(self, node_ids: list[str], depth: int = 2) -> "KnowledgeGraph":
        """Get a subgraph around specific nodes."""
        subgraph = KnowledgeGraph()
        visited = set()

        def expand(nid: str, d: int):
            if d < 0 or nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            node = self.nodes[nid]
            subgraph.add_node(node.label, node.node_type, node.metadata, node.tags, node.id)
            subgraph.nodes[nid].x = node.x
            subgraph.nodes[nid].y = node.y
            subgraph.nodes[nid].color = node.color
            for neighbor in self.adjacency.get(nid, []):
                expand(neighbor, d - 1)
                if neighbor in subgraph.nodes:
                    subgraph.add_edge(nid, neighbor, self._get_edge_weight(nid, neighbor))

        for nid in node_ids:
            expand(nid, depth)
        return subgraph

    # === Export ===

    def to_json(self) -> str:
        """Export graph as JSON."""
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [
                {"source": e.source, "target": e.target, "weight": e.weight, "type": e.edge_type}
                for e in self.edges
            ],
            "communities": [
                {"id": c.id, "nodes": c.nodes, "label": c.label, "color": c.color}
                for c in self.communities
            ],
        }
        return json.dumps(data, indent=2)

    def to_html(self, title: str = "Knowledge Graph") -> str:
        """Export as interactive HTML visualization."""
        nodes_json = json.dumps([n.to_dict() for n in self.nodes.values()])
        edges_json = json.dumps([
            {"source": e.source, "target": e.target, "weight": e.weight}
            for e in self.edges
        ])

        return f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<style>
body {{ margin: 0; background: #0a0a0a; overflow: hidden; font-family: system-ui; }}
#graph {{ width: 100vw; height: 100vh; }}
#controls {{ position: fixed; top: 20px; left: 20px; background: rgba(20,20,30,0.9);
  padding: 16px; border-radius: 12px; color: white; z-index: 10; }}
#controls h3 {{ margin: 0 0 12px 0; font-size: 14px; }}
#search {{ width: 200px; padding: 8px; border-radius: 6px; border: 1px solid #333;
  background: #1a1a2e; color: white; }}
#legend {{ position: fixed; bottom: 20px; right: 20px; background: rgba(20,20,30,0.9);
  padding: 16px; border-radius: 12px; color: white; z-index: 10; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
#info {{ position: fixed; top: 20px; right: 20px; background: rgba(20,20,30,0.9);
  padding: 16px; border-radius: 12px; color: white; z-index: 10; display: none; max-width: 300px; }}
#stats {{ position: fixed; bottom: 20px; left: 20px; background: rgba(20,20,30,0.9);
  padding: 12px; border-radius: 12px; color: #888; font-size: 11px; z-index: 10; }}
</style>
</head>
<body>
<div id="controls">
  <h3>Knowledge Graph</h3>
  <input id="search" placeholder="Search nodes..." />
  <div id="filter" style="margin-top: 8px;"></div>
</div>
<canvas id="graph"></canvas>
<div id="legend"></div>
<div id="info"></div>
<div id="stats"></div>
<script>
const nodes = {nodes_json};
const edges = {edges_json};
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const types = {{memory:'#4a9eff',data:'#7c3aed',skill:'#10b981',mcp:'#f59e0b',
  concept:'#ef4444',file:'#6b7280',tool:'#ec4899',agent:'#8b5cf6'}};

// Build legend
const legend = document.getElementById('legend');
legend.innerHTML = '<div style="font-weight:bold;margin-bottom:8px">Node Types</div>' +
  Object.entries(types).map(([k,v]) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${v}"></div>${k}</div>`
  ).join('');

// Build filter
const filter = document.getElementById('filter');
filter.innerHTML = Object.keys(types).map(t =>
  `<label style="display:block;margin:2px 0;font-size:12px;color:#aaa">
    <input type="checkbox" checked onchange="toggleType('{t}',this.checked)"> {t}</label>`
).join('');

let activeTypes = new Set(Object.keys(types));
let selectedNode = null;
let hoveredNode = null;
let dragging = null;
let offsetX = 0, offsetY = 0;

function toggleType(type, show) {{
  if (show) activeTypes.add(type); else activeTypes.delete(type);
  draw();
}}

function draw() {{
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const visible = nodes.filter(n => activeTypes.has(n.type));

  // Draw edges
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  edges.forEach(e => {{
    const s = visible.find(n => n.id === e.source);
    const t = visible.find(n => n.id === e.target);
    if (s && t) {{
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }}
  }});

  // Draw nodes
  visible.forEach(n => {{
    const r = n.size || 10;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = n.id === selectedNode ? '#fff' :
                    n.id === hoveredNode ? n.color : n.color + 'cc';
    ctx.fill();
    if (n.id === selectedNode) {{
      ctx.strokeStyle = n.color;
      ctx.lineWidth = 3;
      ctx.stroke();
    }}
    ctx.fillStyle = '#fff';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(n.label, n.x, n.y + r + 14);
  }});

  document.getElementById('stats').textContent =
    `Nodes: ${{visible.length}}/${{nodes.length}} | Edges: ${{edges.length}}`;
}}

// Mouse events
canvas.addEventListener('mousedown', e => {{
  const n = nodes.find(n => Math.hypot(n.x - e.clientX, n.y - e.clientY) < n.size + 5);
  if (n) {{ dragging = n; offsetX = e.clientX - n.x; offsetY = e.clientY - n.y; }}
}});
canvas.addEventListener('mousemove', e => {{
  if (dragging) {{ dragging.x = e.clientX - offsetX; dragging.y = e.clientY - offsetY; draw(); }}
  else {{
    hoveredNode = (nodes.find(n => Math.hypot(n.x - e.clientX, n.y - e.clientY) < n.size + 5) || {{}}).id;
    canvas.style.cursor = hoveredNode ? 'pointer' : 'default';
    draw();
  }}
}});
canvas.addEventListener('mouseup', () => {{ dragging = null; }});
canvas.addEventListener('click', e => {{
  const n = nodes.find(n => Math.hypot(n.x - e.clientX, n.y - e.clientY) < n.size + 5);
  if (n) {{
    selectedNode = n.id;
    const info = document.getElementById('info');
    info.style.display = 'block';
    info.innerHTML = `<h3 style="margin:0 0 8px">${{n.label}}</h3>
      <div style="font-size:12px;color:#aaa">Type: ${{n.type}}</div>
      <div style="font-size:12px;color:#aaa">Connections: ${{(n.connections||[]).length}}</div>
      <div style="font-size:12px;color:#aaa">Tags: ${{(n.tags||[]).join(', ')}}</div>`;
    draw();
  }}
}});

// Search
document.getElementById('search').addEventListener('input', e => {{
  const q = e.target.value.toLowerCase();
  nodes.forEach(n => n._match = !q || n.label.toLowerCase().includes(q) || (n.tags||[]).some(t => t.includes(q)));
  draw();
}});

// Animation loop
let time = 0;
function animate() {{
  time += 0.005;
  nodes.forEach(n => {{
    if (n !== dragging) {{
      n.x += Math.sin(time + n.x * 0.01) * 0.1;
      n.y += Math.cos(time + n.y * 0.01) * 0.1;
    }}
  }});
  draw();
  requestAnimationFrame(animate);
}}

animate();
window.addEventListener('resize', () => {{ canvas.width = innerWidth; canvas.height = innerHeight; draw(); }});
</script>
</body>
</html>"""

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": {
                t.value: sum(1 for n in self.nodes.values() if n.node_type == t)
                for t in NodeType
            },
            "avg_connections": (
                sum(len(adj) for adj in self.adjacency.values()) / max(len(self.nodes), 1)
            ),
            "orphan_nodes": len(self.get_orphan_nodes()),
            "hub_nodes": len(self.get_hub_nodes()),
            "communities": len(self.communities),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Knowledge graph CLI")
    parser.add_argument(
        "--file",
        default=None,
        help="JSON file defining the graph (nodes and edges)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Show graph statistics")

    p = sub.add_parser("search", help="Search nodes by query")
    p.add_argument("query", help="Search query")
    p.add_argument(
        "--type",
        default=None,
        choices=[t.value for t in NodeType],
        help="Filter by node type",
    )

    p = sub.add_parser("path", help="Find a path between two nodes")
    p.add_argument("start", help="Start node id")
    p.add_argument("goal", help="Goal node id")
    p.add_argument(
        "--algorithm",
        default="astar",
        choices=["astar", "dijkstra", "bfs", "dfs"],
        help="Pathfinding algorithm",
    )

    sub.add_parser("communities", help="Detect and list communities")

    p = sub.add_parser("layout", help="Run force-directed layout")
    p.add_argument("--iterations", type=int, default=100, help="Layout iterations")

    p = sub.add_parser("export", help="Export the graph")
    p.add_argument("--format", choices=["json", "html"], default="json", help="Export format")
    p.add_argument("--output", default=None, help="Output file path")

    args = parser.parse_args(argv)

    def load_graph(g):
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                spec = json.load(f)
            id_map = {}
            for node in spec.get("nodes", []):
                ntype = NodeType(node.get("type", "concept"))
                nid = g.add_node(
                    node["label"],
                    ntype,
                    metadata=node.get("metadata"),
                    tags=node.get("tags"),
                    node_id=node.get("id"),
                ).id
                id_map[node.get("id", nid)] = nid
            for edge in spec.get("edges", []):
                src = id_map.get(edge["source"], edge["source"])
                tgt = id_map.get(edge["target"], edge["target"])
                g.add_edge(
                    src,
                    tgt,
                    weight=edge.get("weight", 1.0),
                    edge_type=edge.get("type", "related"),
                    directed=edge.get("directed", False),
                )

    try:
        graph = KnowledgeGraph()
        load_graph(graph)

        if args.command == "stats":
            print(json.dumps(graph.get_stats(), indent=2, default=str))
        elif args.command == "search":
            ntype = NodeType(args.type) if args.type else None
            results = graph.search(args.query, node_type=ntype)
            print(json.dumps([n.to_dict() for n in results], indent=2, default=str))
        elif args.command == "path":
            path = getattr(graph, f"find_path_{args.algorithm}")(args.start, args.goal)
            print(json.dumps(
                {"start": args.start, "goal": args.goal, "found": path is not None, "path": path},
                indent=2,
                default=str,
            ))
        elif args.command == "communities":
            comms = graph.detect_communities()
            print(json.dumps(
                [{"id": c.id, "label": c.label, "nodes": c.nodes} for c in comms],
                indent=2,
                default=str,
            ))
        elif args.command == "layout":
            graph.layout_force_directed(iterations=args.iterations)
            print(json.dumps(
                {
                    "positions": {
                        nid: [round(n.x, 2), round(n.y, 2)]
                        for nid, n in graph.nodes.items()
                    }
                },
                indent=2,
            ))
        elif args.command == "export":
            out = graph.to_json() if args.format == "json" else graph.to_html()
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(out)
                print(json.dumps({"written": args.output, "bytes": len(out)}, indent=2))
            else:
                print(out)
        else:
            parser.error("Unknown command")
            return 2
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
