
import sys
import os
import torch
from src.data.ast_parser import parse_code
from src.data.graph_builder import ast_to_graph
from src.models.hybrid_model import CodeFluxModel
from src.utils.visualizer import create_heatmap

def analyze_file(file_path, model):
    try:
        with open(file_path, 'r') as f:
            code = f.read()
    except Exception as e:
        return None

    # Parse
    parsed = parse_code(code)
    if "error" in parsed: return None
    
    # Graph
    graph_data = ast_to_graph(parsed['ast'])
    if not graph_data: return None

    # Features
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones((1, 128))
    stats = parsed['stats']
    features = torch.tensor([[
        float(stats['complexity']),
        float(stats['max_depth']),
        float(stats['num_functions']),
        0.0, 0.0
    ]])
    
    # Predict
    with torch.no_grad():
        score, _, attentions = model(input_ids, attention_mask, graph_data['x'], graph_data['edge_index'], features)
        
    return {
        "file": file_path,
        "score": score.item(),
        "stats": stats,
        "code": code,
        "attentions": attentions
    }

def run_portfolio_analysis(dir_path, model):
    print(f"--- Scanning Directory: {dir_path} ---")
    results = []
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                res = analyze_file(full_path, model)
                if res:
                    results.append(res)
                    print(f"Analyzed {file}: Risk={res['score']:.4f}")
                    
    # Sort by risk
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n--- TOP RISKY FILES ---")
    for r in results[:5]:
        print(f"{r['score']:.4f} | {r['file']}")
        
    # Generate Dashboard for the riskiest file for now
    if results:
        top_risk = results[0]
        print(f"\nGenerating detailed report for top risk file: {top_risk['file']}")
        cls_attn = top_risk['attentions'][0, :, 0, :].mean(dim=0).tolist()
        create_heatmap(top_risk['code'], cls_attn, top_risk['stats'], output_path="risk_report.html")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Load Model
    model = CodeFluxModel(feature_dim=5)
    if os.path.exists("models_saved/codeflux_model.pth"):
        model.load_state_dict(torch.load("models_saved/codeflux_model.pth"))
    model.eval()

    if os.path.isdir(target):
        run_portfolio_analysis(target, model)
    else:
        # Single file mode (reusing logic)
        res = analyze_file(target, model)
        if res:
            print(f"Risk Score: {res['score']:.4f}")
            cls_attn = res['attentions'][0, :, 0, :].mean(dim=0).tolist()
            create_heatmap(res['code'], cls_attn, res['stats'], output_path="risk_report.html")
