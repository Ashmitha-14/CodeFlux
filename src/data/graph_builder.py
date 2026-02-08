
import ast
import networkx as nx
import torch

class GraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_count = 0
        self.node_map = {}  # ast_node -> graph_node_id
        self.node_features = []  # List of feature vectors (mocked for now)

    def _get_node_type_encoding(self, node):
        # Simplified one-hot or embedding lookup Mock
        # In real scenario, map type(node).__name__ to an integer ID
        return hash(type(node).__name__) % 100  # Simple hash for prototype

    def generic_visit(self, node):
        node_id = self.node_count
        self.node_count += 1
        self.node_map[node] = node_id
        
        # Feature extraction
        feature = self._get_node_type_encoding(node)
        self.node_features.append(feature) # Feature vector of size 1 for simplicity

        self.graph.add_node(node_id, type=type(node).__name__)

        super().generic_visit(node)

        # Add edges from parent to children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        if item in self.node_map:
                            child_id = self.node_map[item]
                            self.graph.add_edge(node_id, child_id, type="parent_child")
            elif isinstance(value, ast.AST):
                if value in self.node_map:
                    child_id = self.node_map[value]
                    self.graph.add_edge(node_id, child_id, type="parent_child")

def ast_to_graph(tree):
    builder = GraphBuilder()
    builder.visit(tree)
    
    # Convert to standard PyTorch tensors
    num_nodes = builder.node_count
    if num_nodes == 0:
        return None

    # Node Features: [num_nodes, feature_dim]
    # For now, feature_dim = 1
    x = torch.tensor(builder.node_features, dtype=torch.float).unsqueeze(1) 
    
    # Adjacency Matrix (using edge_index format for compatibility with custom GCN)
    # [2, num_edges]
    edges = list(builder.graph.edges)
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return {
        "x": x,
        "edge_index": edge_index,
        "num_nodes": num_nodes
    }
