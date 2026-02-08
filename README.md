
# CodeFlux 🧠 flux of reliable code
**AI-Powered Code Reliability & Risk Prediction System**

CodeFlux is a prototype **Neuro-Symbolic AI system** that analyzes source code to predict bug-prone regions. It combines a **Transformer** (for semantic understanding) and a **Graph Neural Network** (for structural AST analysis) to identify risky code patterns before deployment.

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=CodeFlux+Dashboard+Preview)

## 🚀 Key Features

- **Hybrid AI Model**: Fuses CodeBERT embeddings with GNN structure embeddings.
- **Risk Heatmaps**: Visualizes "attention" weights to show exactly *where* the risk lies (Red zones).
- **Interactive Dashboard**: Rich HTML report with Risk Distribution, Radar Charts, and Metrics.
- **AI Insights**: Generates natural language explanations (e.g., *"High Cyclomatic Complexity detected in function X"*).
- **Directory Scanning**: Scans entire projects to find the riskiest files.
- **REST API**: Flask-based API for integration into CI/CD pipelines.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ashmitha-14/CodeFlux.git
    cd CodeFlux
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Demo**:
    ```bash
    python3 demo.py example_bad_code.py
    ```
    Dimensions of analysis will be saved to `risk_report.html`.

## 💻 Usage

### CLI Mode
Scan a single file:
```bash
python3 demo.py path/to/file.py
```
Scan a directory (Project Level):
```bash
python3 demo.py .
```

### API Mode
Start the server:
```bash
python3 src/api/app.py
```
Send a request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"code": "def hello(): return 1"}'
```

## 🧠 Architecture

1.  **Input**: Source Code.
2.  **Processing**:
    *   **AST Parser**: Extracts structure and metrics (Complexity, Depth).
    *   **Graph Builder**: Converts AST to Graph Data (`x`, `edge_index`).
3.  **Model**:
    *   **Transformer Encoder**: Processes token sequence.
    *   **GNN Encoder**: Processes graph structure.
    *   **Fusion Layer**: Combines Text + Graph + Scalar Metrics.
4.  **Output**: Risk Score (0-1) + Attention Weights.

## 📄 License
MIT License. Built for research and educational purposes.
