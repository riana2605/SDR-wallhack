import argparse
from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Preluare argumente din linia de comanda
parser = argparse.ArgumentParser(description="Live SDR Spectrogram")
parser.add_argument('--center-freq', type=float, default=100, help="Frecventa centrala in MHz")
parser.add_argument('--bandwidth', type=float, default=.5, help="Latimea de banda in MHz")
parser.add_argument('--gain', type=int, default=50, help="Castigul (Gain)")
args = parser.parse_args()

center_freq = args.center_freq * 1e6
bandwidth = args.bandwidth * 1e6
gain = args.gain

# Sample rate-ul calculat in functie de latimea de banda (Nyquist)
sample_rate = bandwidth * 2

# 1. Configurare BladeRF
sdr = _bladerf.BladeRF()

print("Device info:", _bladerf.get_device_list()[0])
print("libbladeRF version:", _bladerf.version())
print("Firmware version:", sdr.get_fw_version())
print("FPGA version:", sdr.get_fpga_version())

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))

sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500)

bytes_per_sample = 4

# Aplicare parametri pe hardware
rx_ch.frequency = int(center_freq)
rx_ch.sample_rate = int(sample_rate)
rx_ch.bandwidth = int(bandwidth)
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

# Parametri FFT și Buffer
fft_size = 2048
num_rows_display = 100
frames_per_update = 4
num_samples_to_read = fft_size * frames_per_update

buf = bytearray(num_samples_to_read * bytes_per_sample)
waterfall = np.zeros((num_rows_display, fft_size))

# 2. Configurare Interfață Grafică
plt.ioff()
fig, ax = plt.subplots()

f_min = (center_freq - sample_rate / 2) / 1e6
f_max = (center_freq + sample_rate / 2) / 1e6
extent = [f_min, f_max, num_rows_display, 0]

# Setare heatmap pe 'jet' (albastru minim, rosu maxim), ajustare oprita
im = ax.imshow(waterfall, aspect='auto', extent=extent, vmin=-60, vmax=10, origin='lower', cmap='jet')
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Frames (Timp)")
ax.set_title(f"Live SDR Spectrogram (CF: {center_freq / 1e6} MHz)")
fig.colorbar(im, label="Power [dB]")


# 3. Funcția de actualizare
def update_plot(frame):
    global waterfall

    # Citește datele
    sdr.sync_rx(buf, num_samples_to_read)
    samples = np.frombuffer(buf, dtype=np.int16)
    samples = samples[0::2] + 1j * samples[1::2]
    samples /= 2048.0

    # Procesare FFT
    for i in range(frames_per_update):
        segment = samples[i * fft_size:(i + 1) * fft_size]
        fft_data = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(segment))) ** 2 + 1e-12)

        # Shift in-place
        waterfall[:-1, :] = waterfall[1:, :]
        waterfall[-1, :] = fft_data

    # Update grafic
    im.set_data(waterfall)
    return [im]


print(f"Starting live receive (CF: {center_freq} Hz, BW: {bandwidth} Hz, SR: {sample_rate} Hz, Gain: {gain})...")
rx_ch.enable = True

ani = animation.FuncAnimation(fig, update_plot, interval=10, blit=False, cache_frame_data=False)

# 4. Blocarea execuției și clean-up
try:
    plt.show(block=True)
except KeyboardInterrupt:
    print("\nCtrl+C apăsat.")
except Exception as e:
    print(f"\nA apărut o eroare: {e}")
finally:
    rx_ch.enable = False
    print("SDR hardware disabled. Ieșire curată.")
    sys.exit(0)