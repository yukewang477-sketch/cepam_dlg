import matplotlib.pyplot as plt

# Placeholder data
# Replace the placeholder lists with your actual data
epsilon_015 = [1, 2, 3, 4, 5]  # Privacy budget for δ=0.015
# accuracy_cepam_015 = [85, 87, 89, 91, 92]  # Replace with your data
# accuracy_fl_sdq_015 = [84, 86, 88, 90, 91]  # Replace with your data

epsilon_01 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9.9]  # Privacy budget for δ=0.01
# accuracy_cepam_01 = [83, 85, 87, 89, 90, 91, 92, 93, 94, 95]  # Replace with your data
# accuracy_fl_sdq_01 = [82, 84, 86, 88, 89, 90, 91, 92, 93, 94]  # Replace with your data

# Line 1
accuracy_cepam_015 = [84.859, 89.989, 91.67, 92.824, 93.27]

# Line 2
accuracy_fl_sdq_015 = [90.028, 91.527, 92.196, 92.364, 92.418]

# Line 3
accuracy_cepam_01 = [77.531, 84.72, 88.41, 90.097, 91.101, 91.747, 92.448, 92.838, 93.086, 93.25]

# Line 4
accuracy_fl_sdq_01 = [86.14, 90.037, 91.082, 91.668, 92.095, 92.235, 92.293, 92.3901, 92.415, 92.391]
# Create the figure
plt.figure(figsize=(8, 5))

# Plot for δ=0.015
plt.plot(epsilon_015, accuracy_cepam_015, label=r'CEPAM-Gaussian ($\delta=0.015$)', marker='s', linestyle='-', color='green')
plt.plot(epsilon_015, accuracy_fl_sdq_015, label=r'FL+Gaussian+SDQ ($\delta=0.015$)', marker='s', linestyle='--', color='blue')

# Plot for δ=0.01
plt.plot(epsilon_01, accuracy_cepam_01, label=r'CEPAM-Gaussian ($\delta=0.01$)', marker='o', linestyle='-', color='red')
plt.plot(epsilon_01, accuracy_fl_sdq_01, label=r'FL+Gaussian+SDQ ($\delta=0.01$)', marker='o', linestyle='--', color='olive')

# Add labels, grid, and legend
plt.xlabel(r'Privacy $\epsilon$', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
# plt.title('Learning Accuracy and Privacy Trade-off', fontsize=14)
plt.grid(alpha=0.3)
plt.xticks(range(1, 11), fontsize=10)  # Display ticks from 1 to 10
plt.legend(fontsize=10, loc='lower right')

# Save the figure
plt.tight_layout()
plt.savefig('privacy_tradeoff.png', dpi=300)
