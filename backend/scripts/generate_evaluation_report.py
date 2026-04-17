#!/usr/bin/env python3
"""
Full Evaluation Report Generator
Run: python scripts/generate_evaluation_report.py
"""
import os, sys, json
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
MODEL_PATH   = BASE_DIR / 'models' / 'cnn_drowsiness.h5'
PREPARED_DIR = BASE_DIR / 'datasets' / '_prepared'
HISTORY_PATH = BASE_DIR / 'models' / 'checkpoints' / 'training_history.json'
REPORT_DIR   = BASE_DIR / 'models' / 'evaluation_report'
IMAGE_SIZE   = (128, 128)
BATCH_SIZE   = 32

os.makedirs(str(REPORT_DIR), exist_ok=True)
print("=" * 60)
print("Evaluation Report Generator")
print("=" * 60)

# ── Load model ─────────────────────────────────────────────────────────────
print("\nLoading model...")
model = keras.models.load_model(str(MODEL_PATH))
print("OK - Model loaded")

# ── Load test data ─────────────────────────────────────────────────────────
print("Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    str(PREPARED_DIR / 'test'),
    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)
print(f"OK - Test samples: {test_gen.samples}")

# ── Predictions ────────────────────────────────────────────────────────────
print("Running predictions (this takes ~2 mins)...")
test_gen.reset()
y_pred_proba = model.predict(test_gen, verbose=1).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
y_true = test_gen.classes

# ── Metrics ────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=0)
f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec)

FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

# ROC — sklearn returns fpr/tpr of length N+1, thresholds of length N
fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr_arr, tpr_arr)

# Align for FAR/FRR curve — ensure all arrays are the same length
# roc_curve returns fpr/tpr of length N+1, thresholds of length N
# We trim everything to min length to be safe
min_len = min(len(fpr_arr), len(tpr_arr), len(thr_arr))
fpr_t = fpr_arr[:min_len]
tpr_t = tpr_arr[:min_len]
thr_t = thr_arr[:min_len]
fnr_t = 1 - tpr_t

eer_idx = int(np.argmin(np.abs(fpr_t - fnr_t)))
EER = (fpr_t[eer_idx] + fnr_t[eer_idx]) / 2
EER_thr = thr_t[eer_idx]

prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_proba)
avg_prec = average_precision_score(y_true, y_pred_proba)

# Training history
history_data = {}
if HISTORY_PATH.exists():
    with open(str(HISTORY_PATH)) as f:
        history_data = json.load(f)

print("OK - Metrics computed")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main Dashboard (3x3)
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1: Main Dashboard...")
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Driver Drowsiness Detection — Evaluation Dashboard',
             fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1. Confusion Matrix (counts)
ax = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Alert', 'Drowsy'],
            yticklabels=['Alert', 'Drowsy'],
            linewidths=0.5, linecolor='gray')
for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.72, f'({cm_norm[i,j]*100:.1f}%)',
                ha='center', va='center', fontsize=9, color='dimgray')
ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

# 2. Normalized Heatmap
ax = fig.add_subplot(gs[0, 1])
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            xticklabels=['Alert', 'Drowsy'],
            yticklabels=['Alert', 'Drowsy'],
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

# 3. ROC Curve
ax = fig.add_subplot(gs[0, 2])
ax.plot(fpr_arr, tpr_arr, color='darkorange', lw=2,
        label=f'ROC (AUC={roc_auc:.4f})')
ax.plot([0,1],[0,1], 'navy', lw=1, linestyle='--', label='Random')
ax.scatter(fpr_t[eer_idx], tpr_t[eer_idx], color='red', s=80, zorder=5,
           label=f'EER={EER*100:.2f}%')
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3)

# 4. FAR / FRR / EER
ax = fig.add_subplot(gs[1, 0])
ax.plot(thr_t, fpr_t, color='red',  lw=2, label='FAR')
ax.plot(thr_t, fnr_t, color='blue', lw=2, label='FRR')
ax.axvline(x=EER_thr, color='green', linestyle='--', lw=1.5,
           label=f'EER={EER*100:.2f}% @ {EER_thr:.3f}')
ax.set_xlabel('Threshold'); ax.set_ylabel('Error Rate')
ax.set_title('FAR / FRR / EER Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0, 0.5])

# 5. Precision-Recall Curve
ax = fig.add_subplot(gs[1, 1])
ax.plot(rec_curve, prec_curve, color='purple', lw=2,
        label=f'PR (AP={avg_prec:.4f})')
ax.axhline(y=sum(y_true)/len(y_true), color='gray', linestyle='--', label='Baseline')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])

# 6. Confidence Distribution
ax = fig.add_subplot(gs[1, 2])
ax.hist(y_pred_proba[y_true==0], bins=50, alpha=0.6, color='green', label='Alert (True)')
ax.hist(y_pred_proba[y_true==1], bins=50, alpha=0.6, color='red',   label='Drowsy (True)')
ax.axvline(x=0.5,    color='black',  linestyle='--', lw=1.5, label='Threshold=0.5')
ax.axvline(x=EER_thr, color='orange', linestyle='--', lw=1.5, label=f'EER thr={EER_thr:.3f}')
ax.set_xlabel('Predicted Probability (Drowsy)'); ax.set_ylabel('Count')
ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 7. Per-Class Metrics
ax = fig.add_subplot(gs[2, 0])
report = classification_report(y_true, y_pred,
                                target_names=['Alert','Drowsy'],
                                output_dict=True)
classes = ['Alert','Drowsy']
metrics = ['precision','recall','f1-score']
x = np.arange(len(classes)); w = 0.25
for i, m in enumerate(metrics):
    vals = [report[c][m] for c in classes]
    bars = ax.bar(x + i*w, vals, w, label=m.capitalize(),
                  color=['#2196F3','#4CAF50','#FF9800'][i], alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x+w); ax.set_xticklabels(classes)
ax.set_ylim([0.85, 1.02]); ax.set_ylabel('Score')
ax.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# 8. Error Rate Bar
ax = fig.add_subplot(gs[2, 1])
names  = ['FAR\n(False Accept)', 'FRR\n(False Reject)', 'EER\n(Equal Error)']
values = [FAR*100, FRR*100, EER*100]
bars = ax.bar(names, values, color=['#F44336','#2196F3','#FF9800'], alpha=0.85, width=0.5)
for b, v in zip(bars, values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Error Rate (%)'); ax.set_ylim([0, max(values)*1.4])
ax.set_title('Error Rate Metrics', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 9. Summary Table
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
rows = [
    ['Accuracy',       f'{test_acc*100:.2f}%'],
    ['Precision',      f'{test_prec*100:.2f}%'],
    ['Recall',         f'{test_rec*100:.2f}%'],
    ['F1-Score',       f'{f1*100:.2f}%'],
    ['AUC-ROC',        f'{roc_auc:.4f}'],
    ['Avg Precision',  f'{avg_prec:.4f}'],
    ['FAR',            f'{FAR*100:.2f}%'],
    ['FRR',            f'{FRR*100:.2f}%'],
    ['EER',            f'{EER*100:.2f}%'],
    ['Loss',           f'{test_loss:.4f}'],
    ['Test Samples',   f'{test_gen.samples:,}'],
]
tbl = ax.table(cellText=rows, colLabels=['Metric','Value'],
               cellLoc='center', loc='center', colWidths=[0.55,0.45])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.4)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1565C0'); cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#E3F2FD')
    cell.set_edgecolor('white')
ax.set_title('Summary Metrics', fontsize=12, fontweight='bold', pad=10)

fig.savefig(str(REPORT_DIR / 'evaluation_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  OK - evaluation_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training History
# ══════════════════════════════════════════════════════════════════════════
if history_data:
    print("Generating Figure 2: Training History...")
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle('Training History', fontsize=16, fontweight='bold')
    epochs = range(1, len(history_data.get('accuracy', [])) + 1)

    def plot_m(ax, key, title):
        if key in history_data:
            ax.plot(epochs, history_data[key], 'b-o', ms=3, lw=2, label='Train')
        vk = f'val_{key}'
        if vk in history_data:
            ax.plot(epochs, history_data[vk], 'r-s', ms=3, lw=2, label='Validation')
            best = max(history_data[vk]) if key != 'loss' else min(history_data[vk])
            ep   = (history_data[vk].index(best) + 1)
            ax.axvline(x=ep, color='green', linestyle='--', alpha=0.7,
                       label=f'Best={best:.4f}@ep{ep}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plot_m(axes[0,0], 'accuracy',  'Accuracy')
    plot_m(axes[0,1], 'loss',      'Loss')
    plot_m(axes[1,0], 'precision', 'Precision')
    plot_m(axes[1,1], 'recall',    'Recall')
    plt.tight_layout()
    fig2.savefig(str(REPORT_DIR / 'training_history.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("  OK - training_history.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Dataset Distribution
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Dataset Distribution...")
train_gen2 = ImageDataGenerator(rescale=1./255).flow_from_directory(
    str(PREPARED_DIR / 'train'), target_size=IMAGE_SIZE,
    batch_size=1, class_mode='binary', shuffle=False
)
tr_a = int(np.sum(train_gen2.classes == 0))
tr_d = int(np.sum(train_gen2.classes == 1))
te_a = int(np.sum(test_gen.classes == 0))
te_d = int(np.sum(test_gen.classes == 1))

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('Dataset Distribution', fontsize=14, fontweight='bold')
for ax, vals, title in [
    (axes3[0], [tr_a, tr_d], f'Train Set\n({tr_a+tr_d:,} images)'),
    (axes3[1], [te_a, te_d], f'Test Set\n({te_a+te_d:,} images)'),
]:
    ax.pie(vals,
           labels=[f'Alert\n{vals[0]:,}', f'Drowsy\n{vals[1]:,}'],
           colors=['#4CAF50','#F44336'], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize':11})
    ax.set_title(title, fontweight='bold')

cats = ['Train\nAlert','Train\nDrowsy','Test\nAlert','Test\nDrowsy']
vals = [tr_a, tr_d, te_a, te_d]
bars = axes3[2].bar(cats, vals,
                    color=['#81C784','#E57373','#4CAF50','#F44336'], alpha=0.85)
for b, v in zip(bars, vals):
    axes3[2].text(b.get_x()+b.get_width()/2, b.get_height()+200,
                  f'{v:,}', ha='center', va='bottom', fontsize=9)
axes3[2].set_ylabel('Images'); axes3[2].set_title('Split Overview', fontweight='bold')
axes3[2].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig3.savefig(str(REPORT_DIR / 'dataset_distribution.png'),
             dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("  OK - dataset_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Threshold Analysis...")
thrs = np.linspace(0.1, 0.9, 81)
accs, precs, recs, f1s = [], [], [], []
for t in thrs:
    p = (y_pred_proba > t).astype(int)
    tp = np.sum((p==1)&(y_true==1)); tn = np.sum((p==0)&(y_true==0))
    fp = np.sum((p==1)&(y_true==0)); fn = np.sum((p==0)&(y_true==1))
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    accs.append((tp+tn)/len(y_true))
    precs.append(pr); recs.append(rc)
    f1s.append(2*pr*rc/(pr+rc) if (pr+rc)>0 else 0)

best_t = thrs[np.argmax(f1s)]
fig4, ax = plt.subplots(figsize=(12, 6))
ax.plot(thrs, accs,  'b-', lw=2, label='Accuracy')
ax.plot(thrs, precs, 'g-', lw=2, label='Precision')
ax.plot(thrs, recs,  'r-', lw=2, label='Recall')
ax.plot(thrs, f1s,   color='purple', lw=2, label='F1-Score')
ax.axvline(x=0.5,   color='black',  linestyle='--', lw=1.5, label='Default (0.5)')
ax.axvline(x=best_t, color='orange', linestyle='--', lw=1.5, label=f'Best F1 ({best_t:.2f})')
ax.axvline(x=EER_thr, color='cyan',  linestyle='--', lw=1.5, label=f'EER thr ({EER_thr:.3f})')
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.set_title('Metrics vs Decision Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_xlim([0.1, 0.9]); ax.set_ylim([0.5, 1.02])
plt.tight_layout()
fig4.savefig(str(REPORT_DIR / 'threshold_analysis.png'),
             dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print("  OK - threshold_analysis.png")

# ══════════════════════════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FULL EVALUATION SUMMARY")
print("=" * 60)
print(f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Model        : MobileNetV2 + Transfer Learning (CNN)")
print(f"Datasets     : DDD + NTHUDDD + Yawing (3 datasets combined)")
print(f"Train/Test   : 75% / 25%")
print(f"Test Samples : {test_gen.samples:,}")
print(f"\nPERFORMANCE METRICS")
print(f"  Accuracy   : {test_acc*100:.2f}%")
print(f"  Precision  : {test_prec*100:.2f}%")
print(f"  Recall     : {test_rec*100:.2f}%")
print(f"  F1-Score   : {f1*100:.2f}%")
print(f"  AUC-ROC    : {roc_auc:.4f}")
print(f"  Loss       : {test_loss:.4f}")
print(f"\nERROR RATE METRICS")
print(f"  FAR        : {FAR*100:.2f}%  (drowsy missed as alert)")
print(f"  FRR        : {FRR*100:.2f}%  (alert flagged as drowsy)")
print(f"  EER        : {EER*100:.2f}%  @ threshold {EER_thr:.4f}")
print(f"\nCONFUSION MATRIX")
print(f"  TN (Alert correct)   : {TN:,}")
print(f"  FP (Alert as Drowsy) : {FP:,}")
print(f"  FN (Drowsy as Alert) : {FN:,}")
print(f"  TP (Drowsy correct)  : {TP:,}")
print(f"\nREPORT SAVED TO: {REPORT_DIR}")
print(f"  1. evaluation_dashboard.png")
print(f"  2. training_history.png")
print(f"  3. dataset_distribution.png")
print(f"  4. threshold_analysis.png")
print("=" * 60)
