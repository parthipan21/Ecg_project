import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
from azure.iot.device import IoTHubDeviceClient, Message
import time

# Azure IoT Hub connection string
CONNECTION_STRING = "HostName=pqrst.azure-devices.net;DeviceId=comp_port_data;SharedAccessKey=1dGIzcGMQOn1soJw43i4QmeW6yBt1XdS79CJeXbCdhY="  # Replace with your actual connection string

# Create a client for Azure IoT Hub
client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

# Function to send PQRST data to Azure IoT Hub
def send_to_azure(p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, time_in_sec):
    data = {
        "P_peaks": [time_in_sec[p] for p in p_peaks],
        "Q_peaks": [time_in_sec[q] for q in q_peaks],
        "R_peaks": [time_in_sec[r] for r in r_peaks],
        "S_peaks": [time_in_sec[s] for s in s_peaks],
        "T_peaks": [time_in_sec[t] for t in t_peaks],
    }
    message = Message(json.dumps(data))
    client.send_message(message)
    print(f"Sent to Azure IoT Hub: {data}")

# Read ECG data from Excel sheet
excel_file = 'C:\\Users\\parth\\Desktop\\ecg\\Ecc_Raw_Data.xlsx'  # Replace with your file path
df = pd.read_excel(excel_file)

# Assuming the Excel file has columns 'ss' (time in ms) and 'ECG' (ECG values)
time_values = df['Time'].values  # Time values in milliseconds
ecg_signal = df['ECG'].values  # ECG signal values

# Sampling rate (fs = 1 / (3.9 ms) = ~256 Hz)
fs = 256
window_duration = 10  # Duration of the sliding window in seconds
max_samples = fs * window_duration  # Maximum number of samples to keep in memory

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='ECG Signal')
ax.set_ylim(-0, 3000)  # Adjust the y-limits based on the expected signal range
ax.set_xlim(0, 1)  # Start with an initial window size of 10 seconds

plt.title('Real-Time ECG Signal Plot with PQRST Detection')
plt.xlabel('Time (seconds)')
plt.ylabel('ECG Amplitude')

# Add horizontal line at the mean of the ECG signal for reference
mean_ecg = np.mean(ecg_signal)
ax.axhline(y=mean_ecg, color='gray', linestyle='--', linewidth=0.8, label='Mean ECG')

# Plot markers for peaks
r_peaks_plot, = ax.plot([], [], 'ro', label='R-peaks')
q_peaks_plot, = ax.plot([], [], 'go', label='Q-peaks')
s_peaks_plot, = ax.plot([], [], 'bo', label='S-peaks')
p_peaks_plot, = ax.plot([], [], 'mo', label='P-peaks')
t_peaks_plot, = ax.plot([], [], 'co', label='T-peaks')

# BPM text
bpm_text = ax.text(0.05, 0.95, 'BPM: N/A', transform=ax.transAxes)

# Add legend for peaks
ax.legend(loc='upper right')

# Timer for sending data to Azure IoT Hub
last_send_time = time.time()

# Update plot function
def update_plot(ecg_signal, time_values):
    global last_send_time
    
    # Convert time to seconds
    time_in_sec = np.array(time_values) / 1000
    
    # Update the ECG signal line
    line.set_data(time_in_sec, ecg_signal)
    
    # Detect R-peaks (most prominent peaks)
    r_peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.6))  # Assuming 600ms distance between R-peaks

    # Calculate BPM
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs  # R-R intervals in seconds
        average_rr_interval = np.mean(rr_intervals)  # Average R-R interval
        bpm = 60 / average_rr_interval  # Convert to BPM
    else:
        bpm = 0  # Not enough R-peaks to calculate BPM

    # Update BPM text
    bpm_text.set_text(f'BPM: {bpm:.1f}')

    # Detect Q and S peaks (local minima around R-peaks)
    q_peaks = []
    s_peaks = []
    for r_peak in r_peaks:
        q_region = ecg_signal[max(0, r_peak - int(0.1 * fs)):r_peak]
        if len(q_region) > 0:
            q_peaks.append(np.argmin(q_region) + max(0, r_peak - int(0.1 * fs)))

        s_region = ecg_signal[r_peak:r_peak + int(0.1 * fs)]
        if len(s_region) > 0:
            s_peaks.append(np.argmin(s_region) + r_peak)

    # Detect P and T waves (local maxima before and after R-peaks)
    p_peaks = []
    t_peaks = []
    for r_peak in r_peaks:
        p_region = ecg_signal[max(0, r_peak - int(0.3 * fs)):r_peak - int(0.1 * fs)]
        if len(p_region) > 0:
            p_peaks.append(np.argmax(p_region) + max(0, r_peak - int(0.3 * fs)))

        t_region = ecg_signal[r_peak + int(0.1 * fs):r_peak + int(0.4 * fs)]
        if len(t_region) > 0:
            t_peaks.append(np.argmax(t_region) + r_peak + int(0.1 * fs))

    # Update the peak markers
    r_peaks_plot.set_data(time_in_sec[r_peaks], np.array(ecg_signal)[r_peaks])
    q_peaks_plot.set_data(time_in_sec[q_peaks], np.array(ecg_signal)[q_peaks])
    s_peaks_plot.set_data(time_in_sec[s_peaks], np.array(ecg_signal)[s_peaks])
    p_peaks_plot.set_data(time_in_sec[p_peaks], np.array(ecg_signal)[p_peaks])
    t_peaks_plot.set_data(time_in_sec[t_peaks], np.array(ecg_signal)[t_peaks])

    # Send PQRST data to Azure IoT Hub every 5 seconds
    current_time = time.time()
    if current_time - last_send_time >= 5:
        send_to_azure(p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, time_in_sec)
        last_send_time = current_time

    # Draw vertical lines at each second interval for time reference
    for t in range(1, int(np.max(time_in_sec)) + 1):
        ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.5)

    # Adjust x-axis limits to keep showing new data
    ax.set_xlim(0, max(10, time_in_sec[-1]))

    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

# Main loop to plot data continuously
try:
    for i in range(len(ecg_signal)):
        # Update the plot with the current buffer
        update_plot(ecg_signal[:i+1], time_values[:i+1])
except KeyboardInterrupt:
    print("Stopped.")
finally:
    client.disconnect()  # Close Azure IoT Hub connection
