import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DummyClass:
    pass

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'data_schemas':
            return DummyClass
        return super().find_class(module, name)

# Load data
with open('sim_data_occ_bunny_5e09_b_0.7_2sphere.pik', 'rb') as f:
    data = CustomUnpickler(f).load()._data

wfm_data = data['wfm_data']
rx_coords = data['rx_coords']
tx_coords = data['tx_coords']

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# 1. 3D scatter of TX positions
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.scatter(tx_coords[:, 0], tx_coords[:, 1], tx_coords[:, 2], 
            c='red', marker='o', s=1, alpha=0.5)
ax1.set_title('Transmitter Positions')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# 2. 3D scatter of RX positions
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.scatter(rx_coords[:, 0], rx_coords[:, 1], rx_coords[:, 2], 
            c='blue', marker='o', s=1, alpha=0.5)
ax2.set_title('Receiver Positions')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

# 3. Both TX and RX
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.scatter(tx_coords[::100, 0], tx_coords[::100, 1], tx_coords[::100, 2], 
            c='red', marker='o', s=5, alpha=0.5, label='TX')
ax3.scatter(rx_coords[::100, 0], rx_coords[::100, 1], rx_coords[::100, 2], 
            c='blue', marker='o', s=5, alpha=0.5, label='RX')
ax3.set_title('TX & RX Positions (subsampled)')
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
ax3.legend()

# 4. Sample waveform magnitude
ax4 = fig.add_subplot(2, 3, 4)
sample_idx = 0
ax4.plot(np.abs(wfm_data[sample_idx, :]))
ax4.set_title(f'Sample Waveform Magnitude (idx={sample_idx})')
ax4.set_xlabel('Time Sample'); ax4.set_ylabel('Magnitude')
ax4.grid(True)

# 5. Waveform heatmap (first 1000 waveforms)
ax5 = fig.add_subplot(2, 3, 5)
im = ax5.imshow(np.abs(wfm_data[:1000, :]).T, aspect='auto', cmap='viridis')
ax5.set_title('Waveform Magnitude Heatmap (first 1000)')
ax5.set_xlabel('Waveform Index'); ax5.set_ylabel('Time Sample')
plt.colorbar(im, ax=ax5)

# 6. Distribution of waveform magnitudes
ax6 = fig.add_subplot(2, 3, 6)
magnitudes = np.abs(wfm_data).flatten()
ax6.hist(magnitudes[magnitudes > 0], bins=100, log=True)
ax6.set_title('Distribution of Waveform Magnitudes')
ax6.set_xlabel('Magnitude'); ax6.set_ylabel('Count (log scale)')
ax6.grid(True)

plt.tight_layout()
plt.savefig('visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to visualization.png")

# Print statistics
print(f"\nData Statistics:")
print(f"  TX coords range: X[{tx_coords[:, 0].min():.3f}, {tx_coords[:, 0].max():.3f}], "
      f"Y[{tx_coords[:, 1].min():.3f}, {tx_coords[:, 1].max():.3f}], "
      f"Z[{tx_coords[:, 2].min():.3f}, {tx_coords[:, 2].max():.3f}]")
print(f"  RX coords range: X[{rx_coords[:, 0].min():.3f}, {rx_coords[:, 0].max():.3f}], "
      f"Y[{rx_coords[:, 1].min():.3f}, {rx_coords[:, 1].max():.3f}], "
      f"Z[{rx_coords[:, 2].min():.3f}, {rx_coords[:, 2].max():.3f}]")
print(f"  Waveform magnitude range: [{np.abs(wfm_data).min():.3e}, {np.abs(wfm_data).max():.3e}]")
print(f"  Number of waveforms: {wfm_data.shape[0]}")
print(f"  Time samples per waveform: {wfm_data.shape[1]}")

plt.show()