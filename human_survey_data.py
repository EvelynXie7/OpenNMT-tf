import matplotlib.pyplot as plt
import numpy as np

# Human Survey Data, Extracted from https://docs.google.com/spreadsheets/d/1aa7EMr-PRmIXn8wajw2QJq3NucjU65a8-Eb4Un0ZWTI/edit?gid=1823797800#gid=1823797800
algorithms = ['greedy', 'beam', 'inc_beam', 'dec_beam', 'top_k', 'top_p']
means = [3.16, 3.08, 2.56, 2.71, 3.04, 3.44]
std_devs = [1.25, 1.58, 1.50, 1.34, 1.56, 0]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar positions
x_pos = np.arange(len(algorithms))

# Create bars with error bars
bars = ax.bar(x_pos, means, yerr=std_devs, 
              capsize=5, 
              color='steelblue', 
              alpha=0.7,
              edgecolor='black',
              ecolor='black')

# Customize the plot
ax.set_xlabel('Decoding Algorithm', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
ax.set_title('Human Preference Rankings by Decoding Algorithm\n(1=Best, 6=Worst)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, rotation=0)
ax.set_ylim([0, 6])



# Add note about error bars
ax.text(0.02, 0.98, 'Error bars: ±1 SD (n=25)', 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('decoding_rankings.png', dpi=300, bbox_inches='tight')
plt.show()