import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path('figures/.mpl').resolve())

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 1. Figure 3.1: Dataset Split Distribution
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
categories = ['Dev Split\n(Threshold Tuning)', 'Test Split\n(Final Evaluation)']
counts = [100, 400]
colors = ['#3B82F6', '#1E40AF']
bars = ax.bar(categories, counts, color=colors, width=0.45, edgecolor='#1E293B', linewidth=1)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval} images", ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylim(0, 480)
ax.set_ylabel('Number of Images', fontweight='bold')
ax.set_title('Figure 3.1: AccessAI COCO val2017 Dataset Split Distribution', fontweight='bold', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(figures_dir / "fig_3_1_dataset_split.png")
plt.close()

# 2. Figure 3.2: Pipeline Flowchart
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.axis('off')
boxes = [
    ("1. Image Input\nAcquisition", 0.05, 0.5, 0.13, 0.28, '#E0E7FF'),
    ("2. Preprocessing\n(Luminance/Blur)", 0.22, 0.5, 0.14, 0.28, '#EEF2FF'),
    ("3. Parallel Perception\n• YOLO11s (Macro)\n• OWLv2 (Micro)\n• EasyOCR\n• BLIP Caption", 0.40, 0.5, 0.18, 0.42, '#CFFAFE'),
    ("4. Spatial & Claim\nExtraction\n(2D Centroids / NLP)", 0.62, 0.5, 0.16, 0.28, '#FEF3C7'),
    ("5. 5-Tier Verification\n& Safety Gating\n(Consistency Metric)", 0.81, 0.5, 0.16, 0.32, '#DCFCE7'),
]
for text, x, y, w, h, col in boxes:
    rect = patches.FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=col, edgecolor='#334155', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A')

arrows = [(0.18, 0.5, 0.04), (0.36, 0.5, 0.04), (0.58, 0.5, 0.04), (0.78, 0.5, 0.03)]
for x, y, dx in arrows:
    ax.annotate('', xy=(x + dx, y), xytext=(x, y),
                arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.8, headwidth=6, headlength=7))
ax.set_xlim(0, 1.0)
ax.set_ylim(0.2, 0.8)
ax.set_title('Figure 3.2: AccessAI Sequential-Parallel Perception and Verification Architecture', fontweight='bold', fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(figures_dir / "fig_3_2_pipeline_architecture.png")
plt.close()

# 3. Figure 4.1: Component Latency Breakdown
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)
modules = ['Preprocessing', 'Claim Check', 'OCR', 'YOLO11s Detect', 'BLIP Caption', 'Total Pipeline']
latencies = [8.8, 5.1, 169.4, 247.1, 565.0, 573.9]
colors = ['#94A3B8', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6', '#1E293B']
bars = ax.barh(modules, latencies, color=colors, height=0.55, edgecolor='#1E293B', linewidth=1)
for bar in bars:
    wval = bar.get_width()
    ax.text(wval + 12, bar.get_y() + bar.get_height()/2, f"{wval:.1f} ms", ha='left', va='center', fontweight='bold', fontsize=9)
ax.set_xlim(0, 680)
ax.set_xlabel('Execution Time (milliseconds)', fontweight='bold')
ax.set_title('Figure 4.1: Mean Processing Latency by Subsystem (400 Test Images on CUDA)', fontweight='bold', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(figures_dir / "fig_4_1_latency_breakdown.png")
plt.close()

# 4. Figure 5.1: Performance across Scene Conditions
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
conditions = ['Ordinary\n(240 imgs)', 'Cluttered\n(96 imgs)', 'Occluded\n(117 imgs)', 'Low-Light\n(400 imgs)']
x = np.arange(len(conditions))
width = 0.26
precisions = [70.68, 70.69, 69.97, 74.10]
recalls = [72.03, 52.10, 56.64, 55.08]
consistencies = [83.33, 82.59, 81.70, 81.73]

r1 = ax.bar(x - width, precisions, width, label='Precision (%)', color='#2563EB', edgecolor='#1E293B')
r2 = ax.bar(x, recalls, width, label='Recall (%)', color='#F59E0B', edgecolor='#1E293B')
r3 = ax.bar(x + width, consistencies, width, label='Evidence Consistency (%)', color='#10B981', edgecolor='#1E293B')

ax.set_ylabel('Score (%)', fontweight='bold')
ax.set_title('Figure 5.1: Performance Metrics Across Environmental Scene Conditions', fontweight='bold', fontsize=11, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(frameon=True, loc='lower right', fontsize=8.5)
plt.tight_layout()
plt.savefig(figures_dir / "fig_5_1_scene_conditions.png")
plt.close()

# 5. Figure 5.2: Claim Status Distribution
fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
statuses = ['Supported', 'Unverifiable\n(Out of Vocab)', 'Inference\n(Attributes/Verbs)', 'Weak\n(Count/Low Conf)', 'Not Confirmed\n(Hallucination)']
counts = [435, 431, 412, 49, 49]
colors = ['#16A34A', '#64748B', '#8B5CF6', '#EAB308', '#EF4444']
bars = ax.bar(statuses, counts, color=colors, width=0.55, edgecolor='#1E293B', linewidth=1)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 10, str(yval), ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax.set_ylim(0, 500)
ax.set_ylabel('Number of Claims Extracted', fontweight='bold')
ax.set_title('Figure 5.2: Claim Verification Distribution Across 400 Test Captions (1,376 Total Claims)', fontweight='bold', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(figures_dir / "fig_5_2_claim_distribution.png")
plt.close()

# 6. Figure 5.3: Confusion Matrix
fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
matrix = np.array([[427, 8], [14, 35]])
im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
classes = ['Present', 'Absent']
ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]),
       xticklabels=['Declared Supported', 'Declared Not Confirmed'],
       yticklabels=['Ground Truth Present', 'Ground Truth Absent'],
       ylabel='Ground Truth Annotation (COCO)',
       xlabel='System Claim Verification')
ax.set_title('Figure 5.3: Claim Verification Confusion Matrix (Checkable Claims)', fontweight='bold', fontsize=11, pad=12)

thresh = matrix.max() / 2.
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, format(matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(figures_dir / "fig_5_3_confusion_matrix.png")
plt.close()

print("All figures successfully generated in figures/ directory!")
