
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.hybrid_model import CodeFluxModel

def generate_dummy_data(batch_size=4):
    """
    Generates dummy batch data for training.
    """
    # Transformer inputs
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones((batch_size, 128))
    
    # GNN inputs (Simplified: Single large graph or processed as batch)
    # For prototype simplicity, we generate one random graph and reuse it for the batch
    # In a real scenario, we'd batch multiple graphs together into a large disjoint graph
    
    # Random graph with 10 nodes
    # Feature dim = 1 (as updated in graph_builder)
    x = torch.randn(10, 1) 
    edge_index = torch.randint(0, 10, (2, 20)) # 20 edges
    
    # Targets
    target_score = torch.rand(batch_size, 1)
    target_class = torch.randint(0, 3, (batch_size,))
    
    # Handcrafted Features (Batch of 5 scalars: complexity, depth, commits, authors, churn)
    features = torch.randn(batch_size, 5)
    
    return input_ids, attention_mask, x, edge_index, features, target_score, target_class

def train():
    print("Initializing CodeFlux Model...")
    model = CodeFluxModel(feature_dim=5)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_score = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    print("Starting Training Loop (Dummy Data)...")
    model.train()
    
    for epoch in range(5): # 5 epochs for demo
        total_loss = 0
        for batch_idx in range(10): # 10 batches per epoch
            input_ids, attention_mask, gnn_x, gnn_edge_index, features, target_score, target_class = generate_dummy_data()
            
            optimizer.zero_grad()
            
            # Forward pass
            # Note: Current simple GCN implementation takes single graph structure
            # For batch processing, we rely on the expansion logic in FusionModule
            pred_score, pred_class, attentions = model(
                input_ids, 
                attention_mask, 
                gnn_x, 
                gnn_edge_index,
                features
            )
            
            # Combine losses
            loss_score = criterion_score(pred_score, target_score)
            loss_class = criterion_class(pred_class, target_class)
            loss = loss_score + loss_class
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/10:.4f}")
        
    print("Training Complete.")
    
    # Save model
    os.makedirs("models_saved", exist_ok=True)
    torch.save(model.state_dict(), "models_saved/codeflux_model.pth")
    print("Model saved to models_saved/codeflux_model.pth")

if __name__ == "__main__":
    train()
