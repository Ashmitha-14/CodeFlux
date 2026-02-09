
from flask import Flask, request, jsonify, render_template
import sys
import os
import torch

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.ast_parser import parse_code
from src.data.graph_builder import ast_to_graph
from src.models.hybrid_model import CodeFluxModel
from src.utils.visualizer import create_heatmap

app = Flask(__name__, template_folder='templates')

# Initialize model
model = CodeFluxModel(feature_dim=5)
# Use absolute path for robustness or relative from where app is run
model_path = os.path.join(os.path.dirname(__file__), "../../models_saved/codeflux_model.pth")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded trained model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_html', methods=['POST'])
def analyze_html():
    data = request.json
    code = data.get('code')
    
    if not code:
        return jsonify({"error": "No code provided"}), 400

    # 1. Parse AST & Graph
    parsed_ast = parse_code(code)
    graph_data = ast_to_graph(parsed_ast['ast'])
    
    if graph_data is None:
         return jsonify({"error": "Could not build graph. Check code syntax."}), 400

    # 2. Tokenize (Mock)
    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones((1, 128))
    
    # Prepare Features
    stats = parsed_ast['stats']
    features = torch.tensor([[
        float(stats['complexity']),
        float(stats['max_depth']),
        float(stats['num_functions']),
        0.0, 0.0
    ]])

    # 3. Model Inference
    with torch.no_grad():
        risk_score, _, attentions = model(
            input_ids, 
            attention_mask, 
            graph_data['x'], 
            graph_data['edge_index'],
            features
        )
    
    # 4. Generate Heatmap HTML
    cls_attn = attentions[0, :, 0, :].mean(dim=0).tolist()
    html_report = create_heatmap(code, cls_attn, stats, output_path=None)
    
    return jsonify({
        "html_report": html_report,
        "risk_score": float(risk_score.item())
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    code = data.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    # AST Analysis
    parsed = parse_code(code)
    if "error" in parsed:
        return jsonify(parsed), 400
        
    return jsonify({
        "stats": parsed["stats"],
        "message": "Analysis successful"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    code = data.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400

    # 1. Parse AST & Graph
    parsed_ast = parse_code(code)
    graph_data = ast_to_graph(parsed_ast['ast'])
    
    if graph_data is None:
         return jsonify({"error": "Could not build graph"}), 400

    # 2. Tokenize (Mocking tokenizer for now as we don't have tokenizer files)
    # In real app: tokenizer = AutoTokenizer.from_pretrained(...)
    input_ids = torch.randint(0, 1000, (1, 128)) # Dummy sequence
    attention_mask = torch.ones((1, 128))
    
    # Prepare Features
    stats = parsed_ast['stats']
    features = torch.tensor([
        [
            float(stats['complexity']),
            float(stats['max_depth']),
            float(stats['num_functions']),
            0.0, # dummy commits
            0.0  # dummy authors
        ]
    ])

    # 3. Model Inference
    with torch.no_grad():
        risk_score, risk_class_probs, attentions = model(
            input_ids, 
            attention_mask, 
            graph_data['x'], 
            graph_data['edge_index'],
            features
        )
    
    risk_class = torch.argmax(risk_class_probs, dim=1).item()
    risk_labels = ["Low", "Medium", "High"]
    
    return jsonify({
        "risk_score": float(risk_score.item()),
        "risk_class": risk_labels[risk_class],
        "details": parsed_ast["stats"]
    })

@app.route('/explain', methods=['GET'])
def explain():
    # Placeholder for SHAP/Attention explanation
    return jsonify({
        "explanation": "High cyclomatic complexity in function 'main' contributes to risk.",
        "influential_features": ["complexity", "nesting_depth"],
        "attention_score": [0.1, 0.5, 0.9] # Mock
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
