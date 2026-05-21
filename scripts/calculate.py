import numpy as np
from sklearn.metrics import auc, precision_score, recall_score, f1_score

# -------------------------
# INPUT DATA
# -------------------------

scores = np.array([
    0.721, 0.967, 0.678, 0.989, 0.997, 0.824, 0.873, 0.666, 0.977, 0.87,
    0.766, 0.65, 0.591, 0.914, 0.928, 0.773, 0.982, 0.989, 0.858, 0.997,
    0.945, 0.992, 0.752, 0.978, 0.935, 1.0, 0.881, 0.939, 0.601, 0.604,
    0.742
])

all_is_tp = np.zeros_like(scores)   
total_gt = 8

fpi = np.array([
    0.01176471, 0.02352941, 0.03529412, 0.04705882, 0.05882353,
    0.07058824, 0.08235294, 0.09411765, 0.10588235, 0.11764706,
    0.12941176, 0.14117647, 0.15294118, 0.16470588, 0.17647059,
    0.18823529, 0.2, 0.21176471, 0.22352941, 0.23529412, 0.24705882,
    0.25882353, 0.27058824, 0.28235294, 0.29411765, 0.30588235,
    0.31764706, 0.32941176, 0.34117647, 0.35294118, 0.36470588,
    0.37647059
])

tpr = np.zeros_like(fpi)   


# -------------------------
# METRICS
# -------------------------

def interp_recall_at_fpi(target_fpi, fpi, tpr):
    """Interpolate recall (TPR) at a given FPI."""
    return np.interp(target_fpi, fpi, tpr)

R_003 = interp_recall_at_fpi(0.03, fpi, tpr)
R_05  = interp_recall_at_fpi(0.5, fpi, tpr)
R_1   = interp_recall_at_fpi(1.0, fpi, tpr)

# AUC under the FROC curve
AUC = auc(fpi, tpr)

# Precision, Recall, F1 from TP list
precision = precision_score(all_is_tp, np.ones_like(all_is_tp), zero_division=0)
recall = recall_score(all_is_tp, np.ones_like(all_is_tp), zero_division=0)
f1 = f1_score(all_is_tp, np.ones_like(all_is_tp), zero_division=0)

# -------------------------
# PRINT RESULTS
# -------------------------

print(f"R@0.03 = {R_003:.4f}")
print(f"R@0.5  = {R_05:.4f}")
print(f"R@1.0  = {R_1:.4f}")
print(f"AUC    = {AUC:.4f}")

print("\nDetection Metrics:")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")
