
import os
import math
import json
from .ai_explainer import generate_insights

def create_heatmap(code, attention_weights, stats, output_path="risk_report.html"):
    """
    Creates a rich HTML dashboard with code heatmap, charts, and insights.
    """
    
    # --- 1. Data Prep ---
    # Normalize weights
    exp_weights = [math.exp(w * 10) for w in attention_weights]
    sum_exp = sum(exp_weights)
    if sum_exp == 0: sum_exp = 1
    norm_weights = [w / sum_exp for w in exp_weights]
    avg_weight = sum(attention_weights) / len(attention_weights) if attention_weights else 0
    
    # Generate Insights
    insights = generate_insights(stats)
    
    lines = code.split('\n')
    
    # Calculate Risk Distribution for Pie Chart
    high_risk_lines = 0
    med_risk_lines = 0
    low_risk_lines = 0
    
    code_html = ""
    for i, line in enumerate(lines):
        weight = 0
        if len(attention_weights) > 0:
            idx = i % len(attention_weights)
            weight = attention_weights[idx]
            
        ratio = weight / avg_weight if avg_weight > 0 else 1.0
        
        if ratio > 2.0:
            color = f"rgba(255, 59, 48, {min(0.6, (ratio-2)*0.1 + 0.2)})" # Modern Red
            high_risk_lines += 1
        elif ratio > 1.2:
            color = f"rgba(255, 149, 0, {min(0.6, (ratio-1.2)*0.1 + 0.1)})" # Modern Orange
            med_risk_lines += 1
        else:
            color = "rgba(52, 199, 89, 0.05)" # Modern Green
            low_risk_lines += 1
            
        bg_style = f"background-color: {color}"
        code_html += f'<div class="line" style="{bg_style}"><span class="line-num">{i+1}</span>{line}</div>'

    # --- 2. HTML Template (Modern Dark/Light Dashboard) ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeFlux Risk Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #0d1117;
                --card-bg: #161b22;
                --text-main: #c9d1d9;
                --text-secondary: #8b949e;
                --border: #30363d;
                --accent: #58a6ff;
                --red: #ff7b72;
                --orange: #d29922;
                --green: #3fb950;
            }}
            body {{ font-family: 'Inter', sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            
            /* Header */
            header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
            h1 {{ margin: 0; font-size: 24px; color: var(--text-main); }}
            .badge {{ background: var(--accent); color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
            
            /* Grid Layout */
            .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
            
            /* Cards */
            .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
            h2 {{ margin-top: 0; font-size: 16px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }}
            
            /* Code View */
            .code-view {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 13px; line-height: 1.5; overflow-x: auto; }}
            .line {{ padding: 2px 4px; border-radius: 2px; }}
            .line-num {{ color: var(--text-secondary); margin-right: 15px; user-select: none; width: 30px; display: inline-block; text-align: right; }}
            
            /* Metrics grid */
            .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
            .metric-item {{ background: rgba(255,255,255,0.03); padding: 15px; border-radius: 4px; text-align: center; }}
            .metric-val {{ font-size: 24px; font-weight: 700; display: block; }}
            .metric-label {{ font-size: 12px; color: var(--text-secondary); }}
            
            /* Insights List */
            ul.insights {{ list-style: none; padding: 0; }}
            li.insight {{ margin-bottom: 10px; padding: 10px; background: rgba(56, 139, 253, 0.1); border-left: 3px solid var(--accent); font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>CodeFlux Risk Analysis</h1>
                    <span style="color: var(--text-secondary); font-size: 14px;">AI-Powered Code Reliability Report</span>
                </div>
                <div class="badge">v1.0.0 Pro</div>
            </header>
            
            <div class="grid">
                <!-- Left Column: Code Heatmap -->
                <div>
                    <div class="card">
                        <h2>Attention Heatmap</h2>
                        <div class="code-view">
                            {code_html}
                        </div>
                    </div>
                </div>
                
                <!-- Right Column: Analytics -->
                <div>
                    <!-- Summary Metrics -->
                    <div class="card">
                        <h2>Key Metrics</h2>
                        <div class="metrics">
                            <div class="metric-item">
                                <span class="metric-val" style="color: var(--orange)">{stats.get('complexity', 0)}</span>
                                <span class="metric-label">Cyclomatic Complexity</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-val" style="color: var(--accent)">{stats.get('max_depth', 0)}</span>
                                <span class="metric-label">Max Nesting Depth</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-val">{stats.get('num_functions', 0)}</span>
                                <span class="metric-label">Functions</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-val">{len(lines)}</span>
                                <span class="metric-label">Lines of Code</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- AI Insights -->
                    <div class="card">
                        <h2>🤖 AI Insights</h2>
                        <ul class="insights">
                            {''.join([f'<li class="insight">{insight}</li>' for insight in insights])}
                        </ul>
                    </div>
                    
                    <!-- Charts -->
                    <div class="card">
                        <h2>Risk Distribution</h2>
                        <canvas id="riskChart"></canvas>
                    </div>
                     <div class="card">
                        <h2>Code Dimensions</h2>
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Risk Distribution Pie Chart
            const ctxRisk = document.getElementById('riskChart').getContext('2d');
            new Chart(ctxRisk, {{
                type: 'doughnut',
                data: {{
                    labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                    datasets: [{{
                        data: [{high_risk_lines}, {med_risk_lines}, {low_risk_lines}],
                        backgroundColor: ['#ff7b72', '#d29922', '#3fb950'],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#8b949e' }} }} }}
                }}
            }});
            
            // Code Dimensions Radar Chart
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            new Chart(ctxRadar, {{
                type: 'radar',
                data: {{
                    labels: ['Complexity', 'Nesting', 'Length', 'Functions'],
                    datasets: [{{
                        label: 'Current File',
                        data: [
                            Math.min({stats.get('complexity', 0)}, 20), 
                            Math.min({stats.get('max_depth', 0)} * 2, 20), 
                            Math.min({len(lines)} / 10, 20), 
                            Math.min({stats.get('num_functions', 0)} * 2, 20)
                        ],
                        backgroundColor: 'rgba(88, 166, 255, 0.2)',
                        borderColor: '#58a6ff',
                        pointBackgroundColor: '#58a6ff'
                    }}]
                }},
                options: {{
                    scales: {{
                        r: {{
                            angleLines: {{ color: '#30363d' }},
                            grid: {{ color: '#30363d' }},
                            pointLabels: {{ color: '#8b949e' }},
                            ticks: {{ display: false }}
                        }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"Heatmap saved to {output_path}")
    else:
        return html_content
