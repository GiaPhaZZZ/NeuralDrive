import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# --- CONFIGURATION ---
DATA_DIR = "./BCI/data"   # Path to the record EEG.npy files
OUTPUT_DIR = "./BCI/spectrogram_data"   # Output path
SUBFOLDERS = ["long", "short", "extra", "back"]

FS = 250.0                
T_START = 20.0            
T_END = 120.0            

TARGET_SIZE = 224 
DPI = 100
FIG_SIZE = TARGET_SIZE / DPI  

# Ensure root output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting batch EEG spectrogram generation...")
print(f"Targeting a clean 224x224 pixel map for window {T_START}s - {T_END}s.")
print("Skipping all raw files explicitly.\n" + "-"*50)

for folder in SUBFOLDERS:
    input_folder_path = os.path.join(DATA_DIR, folder)
    output_folder_path = os.path.join(OUTPUT_DIR, folder)
    
    # Skip if subfolder doesn't exist in data directory
    if not os.path.exists(input_folder_path):
        print(f"Warning: Subfolder '{folder}' not found in {DATA_DIR}. Skipping.")
        continue
        
    # Ensure corresponding subfolder exists in the output directory
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Find all .npy files
    files = [f for f in os.listdir(input_folder_path) if f.endswith('.npy')]
    print(f"Processing folder '{folder}': Found {len(files)} total file(s).")
    
    for filename in files:
        # --- SKIP RAW FILES ---
        if "raw" in filename.lower():
            continue  # Drops out of the iteration immediately
            
        file_path = os.path.join(input_folder_path, filename)
        
        try:
            # 1. Load data
            raw_data = np.load(file_path)
            
            # 2. Extract dimensions dynamically
            if raw_data.ndim == 1:
                voltage = raw_data
                times = np.arange(len(voltage)) / FS
            elif raw_data.ndim == 2:
                times = raw_data[0, :]
                voltage = raw_data[1, :]
            else:
                print(f" -> [Error] {filename} has invalid dimensions ({raw_data.ndim}). Skipping.")
                continue

            # 3. Slice window (20s to 120s)
            mask = (times >= T_START) & (times <= T_END)
            times_sliced = times[mask]
            voltage_sliced = voltage[mask]

            if len(voltage_sliced) == 0:
                print(f" -> [Warning] {filename} has no data between {T_START}s-{T_END}s. Skipping.")
                continue

            # 4. Compute Spectrogram matrix
            nperseg_val = min(256, len(voltage_sliced) // 2)
            frequencies, segment_times, spec_density = spectrogram(
                voltage_sliced, 
                fs=FS, 
                nperseg=nperseg_val, 
                noverlap=int(nperseg_val * 0.8)
            )

            # 5. Plot and save exactly at 224x224 pixels
            fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI)
            ax = fig.add_axes([0, 0, 1, 1]) # Fills up the entire figure area
            ax.axis('off')                  # Hides axes text, labels, and ticks

            ax.pcolormesh(
                segment_times, 
                frequencies, 
                10 * np.log10(spec_density + 1e-10), 
                shading='gouraud', 
                cmap='jet'
            )
            
            # Keep frequency window clamped to filtered range
            ax.set_ylim(0, min(45, FS / 2))
            ax.set_xlim(segment_times[0], segment_times[-1])

            # Constructing new image filename (.png replaces .npy)
            output_filename = filename.replace('.npy', '.png')
            output_file_path = os.path.join(output_folder_path, output_filename)
            
            # Save matrix cleanly
            plt.savefig(output_file_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
            plt.close(fig) # Liberate memory footprint
            
            print(f" -> Saved: {folder}/{output_filename}")

        except Exception as e:
            print(f" -> [Critical Error] Failed to process {filename}: {e}")

print("-"*50 + "\nBatch job completed successfully!")