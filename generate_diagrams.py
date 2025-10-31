"""
Generate professional architecture diagram for AIOps project
Saves as PNG for GitHub README

Requirements:
pip install matplotlib pillow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create detailed architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'infra': '#E8F5E9',      # Light green
        'data': '#E3F2FD',       # Light blue
        'ml': '#FFF3E0',         # Light orange
        'alert': '#FCE4EC',      # Light pink
        'aws': '#FFF9C4'         # Light yellow
    }
    
    # Title
    ax.text(5, 9.5, 'AIOps: Intelligent Infrastructure Monitoring', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 9.1, 'ML-Powered Anomaly Detection & Predictive Scaling',
            fontsize=12, ha='center', style='italic', color='#666')
    
    # Layer 1: Infrastructure
    infra_box = FancyBboxPatch((0.5, 7.5), 9, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#4CAF50', 
                               facecolor=colors['infra'], 
                               linewidth=2)
    ax.add_patch(infra_box)
    ax.text(5, 8.5, 'Infrastructure Layer', fontsize=14, fontweight='bold', ha='center')
    
    # Servers
    server_positions = [1.5, 3, 4.5, 6, 7.5]
    server_names = ['Web\nServer', 'API\nServer', 'DB\nServer', 'Cache\nServer', 'Worker\nServer']
    
    for i, (pos, name) in enumerate(zip(server_positions, server_names)):
        server = FancyBboxPatch((pos-0.3, 7.7), 0.6, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='#2E7D32',
                               facecolor='white',
                               linewidth=1.5)
        ax.add_patch(server)
        ax.text(pos, 8.0, name, fontsize=8, ha='center', va='center')
    
    # Metrics
    ax.text(9, 8.0, 'Metrics:\n• CPU\n• Memory\n• Network\n• Disk I/O', 
            fontsize=8, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow to Data Layer
    arrow1 = FancyArrowPatch((5, 7.5), (5, 6.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='#666')
    ax.add_artist(arrow1)
    ax.text(5.5, 7.15, '10s intervals', fontsize=8, style='italic', color='#666')
    
    # Layer 2: Data Storage
    data_box = FancyBboxPatch((0.5, 5.5), 9, 1.1,
                             boxstyle="round,pad=0.1",
                             edgecolor='#2196F3',
                             facecolor=colors['data'],
                             linewidth=2)
    ax.add_patch(data_box)
    ax.text(5, 6.4, 'Data Storage Layer', fontsize=14, fontweight='bold', ha='center')
    
    # InfluxDB
    influx = FancyBboxPatch((1.5, 5.7), 2.5, 0.5,
                           boxstyle="round,pad=0.05",
                           edgecolor='#1565C0',
                           facecolor='white',
                           linewidth=1.5)
    ax.add_patch(influx)
    ax.text(2.75, 5.95, 'InfluxDB\nTime-Series Database', fontsize=9, ha='center', va='center')
    
    # Storage info
    ax.text(6, 6.0, '• 10-second granularity\n• 6-month retention\n• 172K points/day\n• Fast queries (<100ms)',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow to ML Layer
    arrow2 = FancyArrowPatch((5, 5.5), (5, 4.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#666')
    ax.add_artist(arrow2)
    ax.text(5.5, 5.15, 'Real-time\nanalysis', fontsize=8, style='italic', color='#666', ha='center')
    
    # Layer 3: ML Pipeline
    ml_box = FancyBboxPatch((0.5, 2.5), 9, 2.1,
                           boxstyle="round,pad=0.1",
                           edgecolor='#FF9800',
                           facecolor=colors['ml'],
                           linewidth=2)
    ax.add_patch(ml_box)
    ax.text(5, 4.4, 'Machine Learning Pipeline', fontsize=14, fontweight='bold', ha='center')
    
    # Feature Engineering
    feature_box = FancyBboxPatch((1, 3.8), 8, 0.5,
                                boxstyle="round,pad=0.05",
                                edgecolor='#F57C00',
                                facecolor='white',
                                linewidth=1.5,
                                linestyle='--')
    ax.add_patch(feature_box)
    ax.text(5, 4.05, 'Feature Engineering: 43 features from 4 base metrics', 
            fontsize=9, ha='center', fontweight='bold')
    ax.text(5, 3.85, 'Rolling Stats (3/5/10) • Temporal • Lag • Rate of Change',
            fontsize=7, ha='center', style='italic')
    
    # Isolation Forest
    iso_forest = FancyBboxPatch((1.5, 2.7), 2.5, 0.9,
                               boxstyle="round,pad=0.05",
                               edgecolor='#E64A19',
                               facecolor='white',
                               linewidth=2)
    ax.add_patch(iso_forest)
    ax.text(2.75, 3.4, 'Anomaly Detector', fontsize=10, ha='center', fontweight='bold')
    ax.text(2.75, 3.15, 'Isolation Forest', fontsize=9, ha='center')
    ax.text(2.75, 2.95, '• 100 trees\n• 85% accuracy\n• 3% false positive', 
            fontsize=7, ha='center')
    
    # LSTM
    lstm_box = FancyBboxPatch((6, 2.7), 2.5, 0.9,
                             boxstyle="round,pad=0.05",
                             edgecolor='#E64A19',
                             facecolor='white',
                             linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(7.25, 3.4, 'LSTM Predictor', fontsize=10, ha='center', fontweight='bold')
    ax.text(7.25, 3.15, 'Time-Series Forecasting', fontsize=9, ha='center')
    ax.text(7.25, 2.95, '• 128→64 units\n• 92% accuracy\n• 30-min forecast',
            fontsize=7, ha='center')
    
    # Arrows to Alert Layer
    arrow3 = FancyArrowPatch((2.75, 2.7), (3.5, 2.0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#666')
    ax.add_artist(arrow3)
    
    arrow4 = FancyArrowPatch((7.25, 2.7), (6.5, 2.0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#666')
    ax.add_artist(arrow4)
    
    # Layer 4: Alerting
    alert_box = FancyBboxPatch((0.5, 0.5), 9, 1.3,
                              boxstyle="round,pad=0.1",
                              edgecolor='#E91E63',
                              facecolor=colors['alert'],
                              linewidth=2)
    ax.add_patch(alert_box)
    ax.text(5, 1.65, 'Intelligent Alerting & Response', fontsize=14, fontweight='bold', ha='center')
    
    # Alert Router
    router = FancyBboxPatch((3.5, 0.9), 3, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor='#C2185B',
                           facecolor='white',
                           linewidth=1.5)
    ax.add_patch(router)
    ax.text(5, 1.35, 'Alert Router', fontsize=10, ha='center', fontweight='bold')
    ax.text(5, 1.1, 'Severity-Based Classification', fontsize=8, ha='center', style='italic')
    
    # Slack
    slack = FancyBboxPatch((1.5, 0.6), 1.2, 0.4,
                          boxstyle="round,pad=0.05",
                          edgecolor='#4A154B',
                          facecolor='white',
                          linewidth=1.5)
    ax.add_patch(slack)
    ax.text(2.1, 0.8, 'Slack\n(All alerts)', fontsize=8, ha='center', va='center')
    
    # PagerDuty
    pagerduty = FancyBboxPatch((7.3, 0.6), 1.2, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor='#06AC38',
                              facecolor='white',
                              linewidth=1.5)
    ax.add_patch(pagerduty)
    ax.text(7.9, 0.8, 'PagerDuty\n(Critical)', fontsize=8, ha='center', va='center')
    
    # Arrows from router
    arrow5 = FancyArrowPatch((3.5, 1.2), (2.7, 0.9),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=1.5, color='#666')
    ax.add_artist(arrow5)
    
    arrow6 = FancyArrowPatch((6.5, 1.2), (7.3, 0.9),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=1.5, color='#666')
    ax.add_artist(arrow6)
    
    # AWS Cloud badge
    aws_badge = FancyBboxPatch((8.5, 0.2), 1.3, 0.3,
                              boxstyle="round,pad=0.03",
                              edgecolor='#FF9900',
                              facecolor=colors['aws'],
                              linewidth=1.5)
    ax.add_patch(aws_badge)
    ax.text(9.15, 0.35, 'Deployed on\nAWS Lambda', fontsize=7, ha='center', va='center')
    
    # Key metrics box
    metrics_text = '''Key Impact Metrics:
• 50% MTTD Reduction (20min → 10min)
• 70% Monitoring Effort Reduction
• 3 Outages Prevented (First Month)
• $2,400/month Cost Savings
• 85% Anomaly Detection Accuracy
• 92% Prediction Accuracy'''
    
    metrics_box = FancyBboxPatch((0.2, 0.05), 3, 0.4,
                                boxstyle="round,pad=0.03",
                                edgecolor='#4CAF50',
                                facecolor='white',
                                linewidth=1.5)
    ax.add_patch(metrics_box)
    ax.text(1.7, 0.25, metrics_text, fontsize=6.5, va='center')
    
    # Tech stack
    tech_text = '''Tech Stack:
Python • TensorFlow • scikit-learn
InfluxDB • Prometheus • AWS Lambda
Terraform • Docker • Slack API'''
    
    tech_box = FancyBboxPatch((3.5, 0.05), 2.5, 0.4,
                             boxstyle="round,pad=0.03",
                             edgecolor='#2196F3',
                             facecolor='white',
                             linewidth=1.5)
    ax.add_patch(tech_box)
    ax.text(4.75, 0.25, tech_text, fontsize=6.5, va='center')
    
    plt.tight_layout()
    plt.savefig('docs/architecture-diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Architecture diagram saved: docs/architecture-diagram.png")
    plt.close()


def create_simple_flow_diagram():
    """Create simplified flow diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'AIOps Data Flow', fontsize=18, fontweight='bold', ha='center')
    
    # Step 1
    step1 = FancyBboxPatch((0.5, 3.5), 1.5, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#4CAF50',
                          facecolor='#E8F5E9',
                          linewidth=2)
    ax.add_patch(step1)
    ax.text(1.25, 4.3, '1. Collect', fontsize=12, ha='center', fontweight='bold')
    ax.text(1.25, 3.9, 'Metrics\nevery 10s', fontsize=9, ha='center')
    
    # Arrow
    ax.annotate('', xy=(2.5, 4.1), xytext=(2.0, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    
    # Step 2
    step2 = FancyBboxPatch((2.5, 3.5), 1.5, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2196F3',
                          facecolor='#E3F2FD',
                          linewidth=2)
    ax.add_patch(step2)
    ax.text(3.25, 4.3, '2. Store', fontsize=12, ha='center', fontweight='bold')
    ax.text(3.25, 3.9, 'InfluxDB\nTime-Series', fontsize=9, ha='center')
    
    # Arrow
    ax.annotate('', xy=(4.5, 4.1), xytext=(4.0, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    
    # Step 3
    step3 = FancyBboxPatch((4.5, 3.5), 1.5, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#FF9800',
                          facecolor='#FFF3E0',
                          linewidth=2)
    ax.add_patch(step3)
    ax.text(5.25, 4.3, '3. Analyze', fontsize=12, ha='center', fontweight='bold')
    ax.text(5.25, 3.9, 'ML Models\nDetect/Predict', fontsize=9, ha='center')
    
    # Arrow
    ax.annotate('', xy=(6.5, 4.1), xytext=(6.0, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    
    # Step 4
    step4 = FancyBboxPatch((6.5, 3.5), 1.5, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#E91E63',
                          facecolor='#FCE4EC',
                          linewidth=2)
    ax.add_patch(step4)
    ax.text(7.25, 4.3, '4. Alert', fontsize=12, ha='center', fontweight='bold')
    ax.text(7.25, 3.9, 'Slack/\nPagerDuty', fontsize=9, ha='center')
    
    # Arrow
    ax.annotate('', xy=(8.5, 4.1), xytext=(8.0, 4.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#666'))
    
    # Step 5
    step5 = FancyBboxPatch((8.5, 3.5), 1.0, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#9C27B0',
                          facecolor='#F3E5F5',
                          linewidth=2)
    ax.add_patch(step5)
    ax.text(9.0, 4.3, '5. Act', fontsize=12, ha='center', fontweight='bold')
    ax.text(9.0, 3.9, 'Auto-\nScale', fontsize=9, ha='center')
    
    # Details below
    details = [
        ("172,800", "data points/day"),
        ("85%", "anomaly accuracy"),
        ("92%", "prediction accuracy"),
        ("<5s", "alert latency"),
        ("50%", "MTTD reduction")
    ]
    
    for i, (value, label) in enumerate(details):
        x = 1 + i * 1.6
        ax.text(x, 2.5, value, fontsize=14, ha='center', fontweight='bold', color='#1976D2')
        ax.text(x, 2.2, label, fontsize=9, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('docs/data-flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Data flow diagram saved: docs/data-flow.png")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('docs', exist_ok=True)
    
    print("Generating architecture diagrams...")
    create_architecture_diagram()
    create_simple_flow_diagram()
    print("\n✓ All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  - docs/architecture-diagram.png (for README)")
    print("  - docs/data-flow.png (simplified version)")
    print("\nAdd to your GitHub repo in the 'docs/' folder")