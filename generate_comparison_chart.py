import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Set high-DPI and modern styling
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

models = ['KNN\n(K-Nearest Neighbors)', 'TF-IDF\nVectorizer', 'Random Forest\nClassifier']
colors = ['#0284c7', '#0f766e', '#8b5cf6']  # Blue, Teal, Purple

# ----------------- SUBPLOT 1: Model Accuracy & Metrics (%) -----------------
ax1 = axes[0]
metrics = ['Top-1 Acc', 'Top-3 Acc', 'Precision', 'Recall', 'F1-Score']
# Data for KNN, TF-IDF, Random Forest
knn_scores = [90.0, 100.0, 89.2, 87.0, 88.1]
tfidf_scores = [86.0, 94.0, 91.5, 88.6, 90.0]
rf_scores = [92.0, 96.0, 92.8, 91.0, 91.9]

x = np.arange(len(metrics))
width = 0.24

b1 = ax1.bar(x - width, knn_scores, width, label='KNN (k=3-5)', color='#0284c7', edgecolor='none', alpha=0.9, zorder=3)
b2 = ax1.bar(x, tfidf_scores, width, label='TF-IDF Vectorizer', color='#0f766e', edgecolor='none', alpha=0.9, zorder=3)
b3 = ax1.bar(x + width, rf_scores, width, label='Random Forest', color='#7c3aed', edgecolor='none', alpha=0.9, zorder=3)

ax1.set_title('Evaluation Metrics Comparison (%)', fontsize=13, fontweight='bold', pad=15, color='#0f172a')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10.5, fontweight='bold', color='#334155')
ax1.set_ylim(70, 105)
ax1.set_ylabel('Score (%)', fontsize=11, fontweight='semibold', color='#334155')
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=9.5)

# Add values above bars
for bar in b1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%' if h < 100 else '100%', ha='center', va='bottom', fontsize=7.5, color='#0369a1', fontweight='bold')

for bar in b2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#0f766e', fontweight='bold')

for bar in b3:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#6d28d9', fontweight='bold')


# ----------------- SUBPLOT 2: Inference Latency (ms) -----------------
ax2 = axes[1]
latencies = [2.15, 1.48, 4.35]  # in ms
bars2 = ax2.bar(models, latencies, color=colors, width=0.45, edgecolor='none', zorder=3)

ax2.set_title('Inference Latency (ms) - Lower is Faster', fontsize=13, fontweight='bold', pad=15, color='#0f172a')
ax2.set_ylabel('Response Time (ms)', fontsize=11, fontweight='semibold', color='#334155')
ax2.set_ylim(0, 5.5)
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.12, f'{h:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1e293b')

# Speed advantage highlight annotation
ax2.annotate('⚡ Fastest\nMatrix Lookup', xy=(1, 1.48), xytext=(1, 2.7),
            arrowprops=dict(facecolor='#0f766e', shrink=0.08, width=1.5, headwidth=6),
            ha='center', fontsize=9, fontweight='bold', color='#0f766e',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdf4', edgecolor='#bbf7d0'))

plt.suptitle('Smart Fridge ML — Candidate Recipe Recommender Models Comparison', fontsize=14, fontweight='bold', y=0.98, color='#0f172a')
plt.tight_layout()

# Save image
out_dir = os.path.join(os.path.dirname(__file__), "presentation")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "model_comparison_chart.png")
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Chart successfully saved at: {out_path}")

static_out = os.path.join(os.path.dirname(__file__), "static", "model_comparison_chart.png")
plt.savefig(static_out, bbox_inches='tight', dpi=300)
print(f"Static copy saved at: {static_out}")
