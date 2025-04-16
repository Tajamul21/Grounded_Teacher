import numpy as np
import matplotlib.pyplot as plt
import os

# Use a serif font for a more academic look
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "figure.dpi": 300
})

# Data
froc_values_our = [0.02197802197802198, 0.06593406593406594, 0.15384615384615385, 0.27472527472527475, 0.31868131868131866, 0.43956043956043955, 0.5714285714285714, 0.6593406593406593, 0.7692307692307693, 0.8571428571428571, 0.9230769230769231, 0.9230769230769231, 0.9230769230769231, 0.9230769230769231]
froc_values_source = [0.01098901098901099, 0.02197802197802198, 0.04395604395604396, 0.08791208791208792, 0.15384615384615385, 0.24175824175824176, 0.45054945054945056, 0.5494505494505495, 0.6813186813186813, 0.7912087912087912, 0.9230769230769231, 0.9230769230769231, 0.9230769230769231, 0.9230769230769231]
fpi_levels = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

# Plot setup
plt.figure(figsize=(8, 5))

# Use edgecolor-only markers for better visibility when overlapping
plt.plot(fpi_levels, froc_values_source, linestyle='-', marker='o', markersize=6,
         markerfacecolor='white', markeredgecolor='blue', color='blue', label="MT")
plt.plot(fpi_levels, froc_values_our, linestyle='-', marker='s', markersize=6,
         markerfacecolor='white', markeredgecolor='red', color='red', label="GT (ours)")

# Axis labels and title
plt.xlabel("False Positives per Image (FPI) — 410 images")
plt.ylabel("True Positive Rate (TPR)")
plt.title("FROC Curve Comparison — DDSM-INB Dataset")

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=True)

output_path = f"dti_froc.png"
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"FROC curve saved at {output_path}")
