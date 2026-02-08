
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class CodeEncoder(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", hidden_dim=768):
        super(CodeEncoder, self).__init__()
        # For prototype, we can use a smaller model or mock if weights are too heavy
        # Using a config to initialize a small random model for demonstration without downloading GBs
        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError:
            # Fallback for offline mode
            config = AutoConfig.from_pretrained("bert-base-uncased")
            
        config.num_hidden_layers = 2 # Lightweight for prototype
        config.output_attentions = True # Enable attention outputs
        self.transformer = AutoModel.from_config(config)
        self.fc = nn.Linear(config.hidden_size, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use simple pooling (CLS token or mean)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Get last layer attention: [batch, num_heads, seq_len, seq_len]
        attentions = outputs.attentions[-1] 
        return self.fc(cls_embedding), attentions

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges] (source, target)
        
        # 1. Self-loops
        num_nodes = x.size(0)
        # Add self loops
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        # 2. Linear Transform
        x = self.linear(x) # [num_nodes, out_features]
        
        # 3. Message Passing (Simple Sum Aggregation)
        # Gather source node features
        src, dst = edge_index
        
        # Create a zero tensor for aggregation
        out = torch.zeros_like(x)
        
        # Scatter add (simplified implementation without scatter_add_)
        # We iterate for simplicity or use index_add_
        out.index_add_(0, dst, x[src])
        
        # Normalize (simplified: divide by degree + 1)
        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(edge_index.size(1), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Renormalize (simplified GCN: D^-0.5 A D^-0.5 X W)
        # This is a very rough approximation of GCN convolution logic
        
        return out

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = SimpleGCNLayer(input_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Global max pooling (simple read-out)
        if x.size(0) == 0:
             return torch.zeros(self.conv2.linear.out_features, device=x.device)
        return torch.max(x, dim=0)[0] # Returns [output_dim]

class FusionModule(nn.Module):
    def __init__(self, transformer_dim, gnn_dim, feature_dim, hidden_dim):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(transformer_dim + gnn_dim + feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # Probability score
        self.risk_classifier = nn.Linear(hidden_dim, 3) # Low, Med, High

    def forward(self, transformer_emb, gnn_emb, features):
        # transformer_emb: [batch, t_dim]
        # gnn_emb: [batch, g_dim] or [g_dim] if single graph
        # features: [batch, f_dim]
        
        if gnn_emb.dim() == 1:
            gnn_emb = gnn_emb.unsqueeze(0).expand(transformer_emb.size(0), -1)
            
        combined = torch.cat((transformer_emb, gnn_emb, features), dim=1)
        x = F.relu(self.fc1(combined))
        
        risk_score = torch.sigmoid(self.fc2(x))
        risk_class = F.softmax(self.risk_classifier(x), dim=1)
        
        return risk_score, risk_class

class CodeFluxModel(nn.Module):
    def __init__(self, transformer_name="microsoft/codebert-base", feature_dim=5):
        super(CodeFluxModel, self).__init__()
        self.code_encoder = CodeEncoder(transformer_name)
        # GNN Input dim = 1 (feature vector size from graph_builder)
        self.gnn_encoder = GNNEncoder(input_dim=1, hidden_dim=64, output_dim=32) 
        # feature_dim matches the number of scalar features (e.g., complexity, depth, churn...)
        self.fusion = FusionModule(transformer_dim=768, gnn_dim=32, feature_dim=feature_dim, hidden_dim=128)

    def forward(self, input_ids, attention_mask, gnn_x, gnn_edge_index, features):
        t_emb, attentions = self.code_encoder(input_ids, attention_mask)
        g_emb = self.gnn_encoder(gnn_x, gnn_edge_index)
        risk_score, risk_class = self.fusion(t_emb, g_emb, features)
        return risk_score, risk_class, attentions
