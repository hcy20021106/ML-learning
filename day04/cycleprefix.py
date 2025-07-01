import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
N = 8               # FFT/IFFT size
L = 3               # Channel length
CP = L - 1          # Cyclic Prefix length

# ----------------------------
# Random OFDM symbol (frequency domain)
# np.random.seed(0)
X = np.random.randn(N) + 1j*np.random.randn(N)

# IFFT to get time domain
x = np.fft.ifft(X)

# Add cyclic prefix: copy last CP samples to the front
x_cp = np.concatenate([x[-CP:], x])

# ----------------------------
# Channel impulse response
h = np.array([0.5, 0.3, 0.2,0,0,1,0,0])

# Linear convolution: transmitted signal through the channel
y = np.convolve(x_cp, h)

# Receiver removes the CP:
y_cp_removed = y[CP : CP + N]

# Circular convolution: should match if CP is long enough
# Use DFT: zero-pad h to length N
h_padded = np.pad(h, (0, N - L))
H = np.fft.fft(h_padded)
Y_freq = np.fft.fft(x) * H
y_circular = np.fft.ifft(Y_freq)

# ----------------------------
# Visualize all steps
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

axs[0].stem(np.arange(N), np.real(x), basefmt=" ", use_line_collection=True)
axs[0].set_title('1. IFFT time-domain signal (no CP)')
axs[0].grid()

axs[1].stem(np.arange(len(x_cp)), np.real(x_cp), basefmt=" ", use_line_collection=True)
axs[1].set_title(f'2. Time-domain with CP (CP length = {CP})')
axs[1].axvspan(0, CP-1, color='orange', alpha=0.3, label='Cyclic Prefix')
axs[1].legend()
axs[1].grid()

axs[2].stem(np.arange(len(y)), np.real(y), basefmt=" ", use_line_collection=True)
axs[2].set_title('3. After channel: linear convolution')
axs[2].axvspan(0, CP-1, color='orange', alpha=0.3, label='CP to discard')
axs[2].legend()
axs[2].grid()

axs[3].stem(np.arange(N), np.real(y_cp_removed), markerfmt='bo', label='Linear conv. + CP removed',
            basefmt=" ", use_line_collection=True)
axs[3].stem(np.arange(N), np.real(y_circular), markerfmt='rx', label='Circular conv.',
            basefmt=" ", use_line_collection=True)
axs[3].set_title('4. Compare: Linear conv. w/ CP removed vs. Circular conv.')
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()

# ----------------------------
# Show numerical error
error = np.linalg.norm(y_cp_removed - y_circular)
print(f"Difference (CP removed vs. circular): {error:.3e}")
