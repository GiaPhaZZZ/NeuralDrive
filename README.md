# ⚡ NEURAL DRIVE v1.0

> **Expanding human automation and cognitive physical augmentation via real-time EEG-BCI classification.**

**Neural Drive** is an open-source Brain-Computer Interface (BCI) framework designed to close the gap between human thought and mechanical execution. By intercepting raw microvolt electroencephalography (EEG) signals, the system leverages a classification pipeline to decode distinct neural signatures directly into physical motor actions — effectively allowing an individual to control machinery with nothing but a thought.

The long-term vision of Neural Drive is to democratize human-robot automation. Imagine a world where human intent is seamlessly mirrored by AI-driven robotics, laying the foundation for advanced cybernetic assistance, physical rehabilitation, and the next frontier of deep-space cosmic exploration.

![EEG Acquisition System](/meta/record_map.png)

---

## 🏎️ Command Matrix & Vehicle Actuation

The current classification model distinguishes between **four neural command vectors**, each mapped to a motor action on the remote vehicle:

| Command | Recording Label | Action | Execution Profile |
| :--- | :--- | :--- | :--- |
| **`FORWARD_1`** | `short` | Short-Distance Pulse | Forward motor rotation for 1 s (precision adjustments). |
| **`FORWARD_2`** | `long` | Long-Distance Stream | Forward motor rotation for 3 s (rapid traversing). |
| **`BACKWARD_1`** | `extra` | Short-Distance Reverse | Reverse motor rotation for 1 s (minor adjustments). |
| **`BACKWARD_2`** | `back` | Long-Distance Reverse | Reverse motor rotation for 3 s (backing out of obstacles). |

The right-hand column (`short` / `long` / `extra` / `back`) is the literal string the model predicts and the folder name used everywhere in the dataset and code — keep this exact spelling in mind when reading the rest of this document.

## 🌌 Core Pillars & Mission Objectives

* 🧠 **Cognitive Automation:** Removing the friction of manual interfaces. You think, and the AI-robot ecosystem handles the physical work.
* 🦾 **Augmenting Human Capability:** Elevating human physical limits through non-invasive neuro-technology, paving the way for tools built for extreme environments and cosmic exploration.
* 🚀 **High-Throughput Telemetry:** Built on top of a specialized 10 kHz stream-ingestion pipeline to ensure near-zero latency from synapse to hardware action.

---

## How thought becomes motion — the end-to-end pipeline

The repository contains five stages that run on three different pieces of hardware (an Arduino UNO, a laptop, and an ESP32-S3). Each stage produces an artifact the next stage consumes:

```
EEG headset/electrodes
        │  analog µV signal
        ▼
arduino/EEG_arduino_record.ino   (Arduino UNO, 10 kHz ADC sampling, USB-serial)
        │  2-byte framed samples over serial (230,400 baud)
        ▼
record/data_acquisition.py       (laptop GUI: live view + 4 s capture + filtering)
        │  one filtered run_N.npy per recording (you sort these by hand into class folders)
        ▼
data_process/prepare_dataset.py  (offline batch job)
        │  one 224×224 spectrogram .png per recording, sorted into class folders
        ▼
train/train.ipynb                (offline training)
        │  eeg_classify.pth  (trained EfficientNetV2-S weights)
        ▼
live.py                          (laptop GUI: live capture + inference + BLE send)
        │  single ASCII command byte '1'-'4' over Bluetooth LE
        ▼
arduino/esp32s3_controller.ino   (ESP32-S3: BLE peripheral + L298N motor driver)
        │  GPIO pulses
        ▼
Motor / vehicle actuation
```

`data_acquisition.py` and `prepare_dataset.py`/`train.ipynb` only need to run once per dataset (to collect trials and train a model). `live.py` is the only script that runs during actual real-time use, and it re-implements the exact same filtering and spectrogram code used during training so that what the model sees in production matches what it was trained on.

---

## Repository map — what each file does

| Path | Runs on | Role |
| :--- | :--- | :--- |
| `arduino/EEG_arduino_record.ino` | Arduino UNO | Bare-metal Timer1 ISR that samples the analog EEG front-end and streams 2-byte-framed samples over serial. |
| `arduino/esp32s3_controller.ino` | ESP32-S3 | BLE GATT peripheral that receives a single command byte and drives an L298N motor driver + status LED accordingly. |
| `record/data_acquisition.py` | Laptop (Python) | Tkinter GUI for connecting to the Arduino, watching the live waveform, and capturing/saving labeled 4-second trials as `.npy`. |
| `data_process/prepare_dataset.py` | Laptop (Python) | Batch script that turns every recorded `.npy` trial into a 224×224 spectrogram `.png`, organized by class. |
| `train/train.ipynb` | Laptop/GPU (Python) | Trains an EfficientNetV2-S classifier on the spectrogram images and saves the best checkpoint, training curves, and a confusion matrix. |
| `live.py` | Laptop (Python) | Real-time GUI: captures a 4 s trial, filters it, builds the spectrogram, classifies it with the trained model, and (optionally) auto-sends the resulting command over BLE to the ESP32-S3. |
| `meta/record_map.png` | — | Reference wiring diagram: BioAmp EXG Pill + electrodes → Arduino UNO → USB → laptop. |

---

## Hardware layer

The acquisition side (see `meta/record_map.png`) is an Arduino UNO paired with an Upside Down Labs **BioAmp EXG Pill** front-end amplifier, feeding 1–3 dry electrodes worn on the scalp/forehead. The Pill's output lands on the Uno's analog input `A0`.

`EEG_arduino_record.ino` configures Timer1 to fire an interrupt roughly every 100 µs (`interrupt_Number = 198` → ~10 kHz with one channel), reads `analogRead(A0)` inside the interrupt, and packs each 10-bit ADC value into **two serial bytes**: the first byte has its top bit set to `1` (frame marker) and carries the upper bits, the second byte has its top bit `0` and carries the lower 7 bits. `loop()` just drains this circular buffer out over `Serial.write()` as fast as it fills. The sketch also accepts inline serial commands (`c:N;` to set channel count, `p:1;` to fire a sync pulse on pin 5) — only the channel-count command is currently used by the Python side (`c:1;`, i.e. single channel). The sketch supports up to 6 channels (`A0`-`A5`) in hardware/firmware even though the rest of the pipeline currently assumes one.

The actuation side is an **ESP32-S3** running `esp32s3_controller.ino`. It exposes one BLE service/characteristic (`SERVICE_UUID` / `CHARACTERISTIC_UUID`, both hardcoded), drives an L298N H-bridge through pins `IN3`/`IN4` (forward/backward only — no `ENA` PWM pin is wired, so the motor always runs at full speed), and lights a single NeoPixel a different color per command. Each command (`'1'`–`'4'`) blocks for a fixed `delay()` (1 s or 3 s) before returning to idle; the motor is force-stopped if the BLE client disconnects mid-command.

---

![Remote Control System](/meta/controller_map.png)

## Stage 1 — Data acquisition (`record/data_acquisition.py`)

A Tkinter + Matplotlib GUI that:

1. Lists serial ports and connects at 230,400 baud, sending `c:1;` to put the Arduino in single-channel mode.
2. Parses the incoming 2-byte frames (`parse_byb_stream`) and converts raw ADC counts to microvolts, centered on the 9-bit ADC midpoint and scaled by the BioAmp gain (`raw_to_uv`, using `ADC_MAX=511`, `ADC_VREF=5.0`, `BYB_GAIN=974.0`).
3. Continuously runs a **stateful causal filter chain** (`LiveFilterPipeline`: 1 Hz high-pass → 40 Hz low-pass → 50/60 Hz notch, all biquad IIR with state preserved across GUI ticks) purely for the rolling 2-second live display.
4. On "START RECORD", buffers `N_CAPTURE = 41,000` raw samples (4 s of data plus 500-sample padding on each side), then runs `finalize_filter`: a **zero-phase** (`sosfiltfilt`/`filtfilt`) version of the same filter chain — DC removal, 1 Hz high-pass, 40 Hz low-pass, notch — and trims the 500-sample padding from each end so the saved array is exactly `N_RECORD = 40,000` samples (4 s at 10 kHz). The padding exists specifically to absorb the IIR warm-up transient that would otherwise corrupt the start of the trial.
5. Shows a save/discard popup; "SAVE" writes the finalized 1-D µV array to `run.npy`, auto-incrementing to `run_1.npy`, `run_2.npy`, … if the base name already exists.

**Important manual step:** the recorder GUI has no class-selection control. After saving, you are responsible for moving/renaming each `run_N.npy` into the correct class subfolder yourself (see dataset layout below) — the script does not know or ask which of the four gestures you just performed.

---

## Dataset layout expected by the rest of the pipeline

`prepare_dataset.py` (and therefore everything downstream) expects recordings to already be sorted like this:

```
BCI/data/
├── long/    *.npy   (FORWARD_2 trials)
├── short/   *.npy   (FORWARD_1 trials)
├── extra/   *.npy   (BACKWARD_1 trials)
└── back/    *.npy   (BACKWARD_2 trials)
```

Any filename containing the substring `raw` is explicitly skipped, so reserve that substring for any raw/unfiltered backups you want to keep alongside the processed trials without them leaking into the dataset.

---

## Stage 2 — Spectrogram generation (`data_process/prepare_dataset.py`)

For every `.npy` file in `BCI/data/<class>/`, the script:

1. Loads the array (handles both a plain 1-D voltage array and a 2-D `[times, voltage]` array).
2. Slices the window `T_START=20 s` to `T_END=120 s` using `FS=250`.
3. Computes a spectrogram (`scipy.signal.spectrogram`, `nperseg=min(256, len//2)`, 80% overlap).
4. Renders it as a borderless `224×224` px image (`jet` colormap, log power in dB, frequency axis clipped to 0–45 Hz) and saves it to `BCI/spectrogram_data/<class>/<same filename>.png`.

### ⚠️ Read this before changing `FS`, `RECORD_SEC`, or the window constants

There is a **scale mismatch** baked into this stage that is important to understand rather than "fix" blindly:

- The hardware/`data_acquisition.py`/`live.py` actually sample and store data at **`FS = 10,000 Hz`**, and each trial is only **4 seconds long** (40,000 samples).
- `prepare_dataset.py` treats every loaded array as if it were sampled at **`FS = 250 Hz`** and slices a **20 s–120 s** window — which, applied to a 40,000-sample array, just means "samples 5,000 to 30,000" (i.e., roughly the 0.5 s–3.0 s portion of the 4 s trial), not real seconds 20–120.
- `live.py`'s `eeg_to_spectrogram_image()` uses the identical constants (`SPEC_FS=250`, `T_START=20`, `T_END=120`) on the identical 40,000-sample buffer, so **training and inference are internally consistent with each other** — the model is trained on, and later fed, the same mis-scaled slice/frequency-axis interpretation. That's why the system still works end-to-end despite the labels being "wrong" in an absolute sense.
- The risk is only if you change one of `FS`, `RECORD_SEC`, `SPEC_FS`, `T_START`, or `T_END` in **one** file without mirroring the change in the other (`prepare_dataset.py` ↔ `live.py`/`data_acquisition.py`). Treat these as a matched set across all three files, not independent per-file options.

---

## Stage 3 — Model training (`train/train.ipynb`)

A single-cell notebook that:

- Scans `BCI/spectrogram_data/{extra,short,long,back}/*.png` and builds a stratified **70 / 10 / 20** train/val/test split (`sklearn.train_test_split`, seeded).
- Applies only `Resize(224,224) → ToTensor → Normalize(ImageNet mean/std)` — **no data augmentation**.
- Builds the model from `torchvision.models.efficientnet_v2_s` pretrained on ImageNet, replacing the classifier head with `Dropout(0.3) → Linear(in_features, 4)`.
- Trains with `AdamW` (`lr=1e-4`, `weight_decay=1e-4`) and `CosineAnnealingLR` for **`NUM_EPOCHS = 5`** epochs, batch size 16, saving the checkpoint whenever validation accuracy improves.
- Outputs: `eeg_classify.pth` (best weights), `history.json`, `training_curves.png` (loss/accuracy/LR), and `test_evaluation.png` (classification report + confusion matrix on the held-out 20%).

`NUM_EPOCHS=5` is a quick-demo setting, not a tuned value — it is one of the first things to raise if accuracy is unsatisfactory (see the quick-reference table below).

---

## Stage 4 — Real-time inference + motor control (`live.py`)

This is the script you actually run during live operation. It combines:

- **Serial reader** — identical framing/parsing/µV-conversion/live-filter code as `data_acquisition.py`.
- **Recorder** — identical 4 s capture + zero-phase `finalize_filter` as `data_acquisition.py`.
- **Spectrogram builder** (`eeg_to_spectrogram_image`) — same slicing/plotting logic as `prepare_dataset.py`, but rendered to an in-memory PIL image instead of a file, then flipped vertically (low frequency at the bottom) and resized to `224×224`.
- **Model loader** (`load_model`/`build_model`) — rebuilds the same `efficientnet_v2_s` architecture (this time with `weights=None`, since the trained checkpoint will overwrite everything anyway) and loads `eeg_classify.pth`.
- **Classifier** (`classify`) — applies the inference-time transform (`Resize → ToTensor → Normalize`, same stats as training) and returns the predicted class plus the full softmax distribution over `CLASSES = ["extra", "short", "long", "back"]`.
- **BLEManager** — runs its own `asyncio` event loop in a background thread (via `bleak`) so BLE I/O never blocks the Tkinter main loop; exposes thread-safe `connect()`/`disconnect()`/`send_command()`.

Operating flow inside the GUI: connect to the EEG serial port → connect to the ESP32 over BLE → click **RECORD & CLASSIFY** → after 4 s the spectrogram and class probabilities appear → if "Auto-send to motor" is checked, the corresponding command byte (`CLASS_TO_CMD`) is written to the BLE characteristic immediately; otherwise the **SEND TO MOTOR** button becomes active for a manual confirm-then-send.

The default BLE target (`BLE_ADDRESS = "E0:72:A1:AA:13:85"`) is the MAC address of one specific ESP32-S3 board and **must be changed** to match your own board (or passed via `--ble`), and `BLE_CHAR_UUID` must always match the `CHARACTERISTIC_UUID` compiled into `esp32s3_controller.ino`.

---

## Quick reference — where to change what

| What you want to change | Edit this | Notes |
| :--- | :--- | :--- |
| Sample rate / trial length / edge padding | `FS`, `RECORD_SEC`, `EDGE_TRIM` in `record/data_acquisition.py` **and** `live.py` | Must match each other; see the sampling-rate warning above before also touching `prepare_dataset.py`'s `FS`/window constants. |
| Notch filter default (mains hum) | `notch_hz` default in `LiveFilterPipeline(...)` / the GUI's "NOTCH" dropdown in either Python GUI | Use 50 Hz (EU/Asia) or 60 Hz (US) depending on your mains frequency. |
| Spectrogram appearance (colormap, freq range, image size) | `prepare_dataset.py` (`cmap`, `ax.set_ylim`, `TARGET_SIZE`) **and** `eeg_to_spectrogram_image` in `live.py` | Keep both in sync — the model is trained on whatever `prepare_dataset.py` produces. |
| Number of EEG channels | `numberOfChannels` logic in `EEG_arduino_record.ino`; the `c:1;` command sent in `data_acquisition.py`/`live.py` | Hardware/firmware supports up to 6 channels; the Python parsing/`raw_to_uv` currently assumes one. |
| Class set / labels | `SUBFOLDERS` in `prepare_dataset.py`, `CLASSES` in `train.ipynb`, `CLASSES`/`CLASS_TO_CMD` in `live.py` | All three lists must stay in the same order and spelling, since the model's output index is just a position in this list. |
| Model architecture / hyperparameters | `build_model`, `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, `WEIGHT_DECAY` in `train.ipynb` | `NUM_EPOCHS=5` is a minimal demo value; raise it (and consider adding augmentation) for a production-quality model. |
| Train/val/test split ratios | `split_samples()` in `train.ipynb` | Currently 70/10/20, stratified by class. |
| BLE device address / characteristic | `BLE_ADDRESS`, `BLE_CHAR_UUID` in `live.py`; `SERVICE_UUID`, `CHARACTERISTIC_UUID` in `esp32s3_controller.ino` | UUIDs must match exactly between the ESP32 sketch and `live.py`. |
| Motor command → action mapping / timing | the `switch(currentCommand)` block in `esp32s3_controller.ino`; `CLASS_TO_CMD` in `live.py` | Durations are hardcoded `delay()` calls; commands are blocking (a new command sent mid-`delay()` will arrive but the switch won't execute until the current one finishes). |
| Serial port baud rate | `Serial.begin(230400)` in `EEG_arduino_record.ino`; the `serial.Serial(port, 230400, ...)` calls in both Python GUIs | Must match on both ends. |
| Dataset/output paths | `DATA_DIR`, `OUTPUT_DIR` in `prepare_dataset.py`; `DATA_ROOT`, `SAVE_PATH`, `HISTORY_PATH`, `PLOT_PATH`, `EVAL_PATH` in `train.ipynb` | All relative to wherever you launch the script from. |

---

## Setup & run order

1. Wire the BioAmp EXG Pill + electrodes to the Arduino UNO as shown in `meta/record_map.png`, and flash `arduino/EEG_arduino_record.ino` to it.
2. Wire the L298N motor driver and NeoPixel to an ESP32-S3 as defined in `arduino/esp32s3_controller.ino`, flash it, and note its BLE MAC address.
3. Install the Python dependencies (see below) and run `record/data_acquisition.py`. Perform each of the four mental/gesture tasks repeatedly, saving a trial each time, then manually sort the resulting `.npy` files into `BCI/data/{long,short,extra,back}/`.
4. Run `data_process/prepare_dataset.py` to convert every trial into a spectrogram under `BCI/spectrogram_data/{long,short,extra,back}/`.
5. Run `train/train.ipynb` to train the classifier; it will produce `eeg_classify.pth` plus training/evaluation plots.
6. Run `live.py --model path/to/eeg_classify.pth --ble <YOUR_ESP32_MAC>`, connect to the EEG serial port and the ESP32 over BLE, and start classifying trials live.

## Dependencies

Python: `numpy`, `scipy`, `matplotlib`, `pillow`, `torch`, `torchvision`, `scikit-learn`, `seaborn`, `pyserial`, `bleak`, `tkinter` (usually bundled with Python).

Arduino libraries: `BLEDevice`/`BLEServer`/`BLEUtils` (built into the ESP32 Arduino core) and `Adafruit_NeoPixel`.