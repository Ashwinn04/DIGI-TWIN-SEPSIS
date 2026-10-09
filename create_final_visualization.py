"""
Create final performance visualization for the sepsis prediction models
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load the analysis results
with open('/Users/admin/Downloads/Capstone/outputs/real_data_analysis.json', 'r') as f:
    data = json.load(f)

# Extract model performance
models = list(data['model_performance'].keys())
aurocs = [data['model_performance'][model]['auroc'] for model in models]

# Create visualization
plt.figure(figsize=(12, 8))

# Bar chart
bars = plt.bar(models, aurocs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Sepsis Prediction Model Performance\nReal ICU Data (546,123 records, 14,057 patients)', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('AUROC Score', fontsize=12)
plt.xlabel('Deep Learning Models', fontsize=12)

# Add value labels on bars
for bar, auroc in zip(bars, aurocs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{auroc:.4f}', ha='center', va='bottom', fontweight='bold')

# Highlight best model
best_idx = np.argmax(aurocs)
bars[best_idx].set_color('#FFD700')  # Gold color for best model
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(2)

# Add horizontal line for random performance
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
plt.legend()

# Add dataset info
plt.text(0.02, 0.98, f'Dataset: {data["dataset_analysis"]["total_records"]:,} records\n'
                     f'Patients: {data["dataset_analysis"]["total_patients"]:,}\n'
                     f'Sepsis Rate: {data["dataset_analysis"]["sepsis_rate"]:.1%}\n'
                     f'Features: {data["dataset_analysis"]["features_count"]}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.grid(True, alpha=0.3)
plt.ylim(0.3, 0.7)
plt.tight_layout()

# Save the plot
output_path = '/Users/admin/Downloads/Capstone/outputs/model_performance_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Performance comparison saved to: {output_path}")

plt.show()
