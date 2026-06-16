"""
live.py  —  EEG Live Classifier + BLE Motor Controller
================================================================
Records 4 s from BYB SpikerShield, classifies the EEG signal using
EfficientNetV2-S, then sends the result over BLE to an ESP32-S3
which drives a motor accordingly.

Class → BLE command → motor action:
  long   → '1' → forward 3 s   (green LED)
  short  → '2' → forward 1 s   (blue LED)
  back   → '3' → backward 3 s  (red LED)
  extra  → '4' → backward 1 s  (yellow LED)

Usage
-----
python live_classify.py --model best_model.pth [--port COM3] [--notch 50]
                        [--ble E0:72:A1:AA:13:85]

Dependencies
------------
    pip install pyserial numpy scipy matplotlib Pillow torch torchvision bleak
"""

import argparse
import asyncio
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import spectrogram as scipy_spectrogram
from PIL import Image

import serial
import serial.tools.list_ports

import torch
import torch.nn as nn
from torchvision import transforms, models

from bleak import BleakClient


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FS          = 10_000
RECORD_SEC  = 4
EDGE_TRIM   = 500
N_RECORD    = FS * RECORD_SEC
N_CAPTURE   = N_RECORD + 2 * EDGE_TRIM
N_LIVE      = FS * 2

ADC_MAX     = 511
ADC_VREF    = 5.0
BYB_GAIN    = 974.0

CLASSES     = ["extra", "short", "long", "back"]
IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

SPEC_FS = 250.0
T_START = 20.0
T_END   = 120.0

# BLE
BLE_ADDRESS   = "E0:72:A1:AA:13:85"
BLE_CHAR_UUID = "87654321-4321-4321-4321-cba987654321"

# EEG class → ESP32 command character
# long   → '1'  forward 3 s  (green)
# short  → '2'  forward 1 s  (blue)
# back   → '3'  backward 3 s (red)
# extra  → '4'  backward 1 s (yellow)
CLASS_TO_CMD = {
    "long":  b"1",
    "short": b"2",
    "back":  b"3",
    "extra": b"4",
}


# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0D0F14",
    "panel":   "#13161E",
    "border":  "#1E2330",
    "accent":  "#00C8FF",
    "accent2": "#7C3AED",
    "warn":    "#F59E0B",
    "ok":      "#10B981",
    "danger":  "#EF4444",
    "text":    "#E2E8F0",
    "muted":   "#64748B",
}

CLASS_COLORS = {
    "extra": "#F59E0B",
    "short": "#00C8FF",
    "long":  "#10B981",
    "back":  "#7C3AED",
}


# ─────────────────────────────────────────────────────────────────────────────
# BLE MANAGER  (asyncio loop in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
class BLEManager:
    """
    Runs an asyncio event loop in a dedicated thread.
    The GUI calls send_command(cmd_bytes) from the main thread; the call is
    thread-safe and non-blocking.
    """
    def __init__(self, address: str, char_uuid: str):
        self.address   = address
        self.char_uuid = char_uuid
        self._client: BleakClient | None = None
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.connected = False
        self.status_cb = None   # optional callable(str, bool) for UI updates

    # ── internal ─────────────────────────────────────────────────────────────
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        """Schedule a coroutine on the BLE event loop from any thread."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _notify(self, msg: str, ok: bool):
        if self.status_cb:
            self.status_cb(msg, ok)

    # ── public API (called from main/GUI thread) ──────────────────────────────
    def connect(self):
        self._submit(self._connect())

    def disconnect(self):
        self._submit(self._disconnect())

    def send_command(self, cmd: bytes):
        """Fire-and-forget: send one command byte to the ESP32."""
        if not self.connected:
            print(f"[BLE] Not connected — command {cmd} dropped.")
            return
        self._submit(self._write(cmd))

    # ── async internals ───────────────────────────────────────────────────────
    async def _connect(self):
        try:
            self._notify(f"Connecting to {self.address}…", False)
            self._client = BleakClient(self.address)
            await self._client.connect()
            self.connected = True
            self._notify(f"BLE ● {self.address}", True)
            print(f"[BLE] Connected to {self.address}")
        except Exception as e:
            self.connected = False
            self._notify(f"BLE ✗ {e}", False)
            print(f"[BLE] Connection error: {e}")

    async def _disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
        self.connected = False
        self._notify("BLE ● Disconnected", False)
        print("[BLE] Disconnected")

    async def _write(self, cmd: bytes):
        try:
            await self._client.write_gatt_char(self.char_uuid, cmd, response=True)
            print(f"[BLE] Sent: {cmd}")
        except Exception as e:
            self.connected = False
            self._notify(f"BLE write error: {e}", False)
            print(f"[BLE] Write error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BYB IIR BIQUAD FILTER
# ─────────────────────────────────────────────────────────────────────────────
class BiquadFilter:
    def __init__(self):
        self.b = np.zeros(3)
        self.a = np.zeros(2)
        self.z = np.zeros(2)

    def _vars(self, Fc, Q, sr):
        w = 2 * np.pi * Fc / sr
        s, c = np.sin(w), np.cos(w)
        return s, c, s / (2 * Q)

    @classmethod
    def highpass(cls, Fc, Q, sr):
        f = cls(); _, c, a = f._vars(Fc, Q, sr); a0 = 1 + a
        f.b = np.array([(1+c)/2, -(1+c), (1+c)/2]) / a0
        f.a = np.array([-2*c, (1-a)]) / a0
        return f

    @classmethod
    def lowpass(cls, Fc, Q, sr):
        f = cls(); _, c, a = f._vars(Fc, Q, sr); a0 = 1 + a
        f.b = np.array([(1-c)/2, (1-c), (1-c)/2]) / a0
        f.a = np.array([-2*c, (1-a)]) / a0
        return f

    @classmethod
    def notch(cls, Fc, Q, sr):
        f = cls(); _, c, a = f._vars(Fc, Q, sr); a0 = 1 + a
        f.b = np.array([1, -2*c, 1]) / a0
        f.a = np.array([-2*c, (1-a)]) / a0
        return f

    def process(self, x: np.ndarray) -> np.ndarray:
        import scipy.signal as ss
        y, self.z = ss.lfilter(self.b, np.r_[1, self.a], x, zi=self.z)
        return y

    def reset(self):
        self.z = np.zeros(2)


class LiveFilterPipeline:
    def __init__(self, sr=FS, notch_hz=50.0):
        self.sr  = sr
        self.hp  = BiquadFilter.highpass(1.0,  0.707, sr)
        self.lp  = BiquadFilter.lowpass(40.0,  0.707, sr)
        self._build_notch(notch_hz)

    def _build_notch(self, hz):
        self.notch_hz = hz
        self._notch   = BiquadFilter.notch(hz, 30.0, self.sr) if hz > 0 else None

    def process(self, raw_int):
        uv = raw_to_uv(raw_int)
        uv = self.hp.process(uv)
        uv = self.lp.process(uv)
        if self._notch:
            uv = self._notch.process(uv)
        return uv

    def set_notch(self, hz):
        self._build_notch(hz)

    def reset(self):
        self.hp.reset(); self.lp.reset()
        if self._notch: self._notch.reset()


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_byb_stream(buf: bytearray):
    samples, i = [], 0
    while i < len(buf) - 1:
        b0, b1 = buf[i], buf[i+1]
        if (b0 & 0x80) and not (b1 & 0x80):
            samples.append(((b0 & 0x03) << 7) | (b1 & 0x7F))
            i += 2
        else:
            i += 1
    return samples, buf[i:]


def raw_to_uv(raw):
    return ((np.asarray(raw, dtype=np.float64) - ADC_MAX / 2.0)
            / ADC_MAX * ADC_VREF / BYB_GAIN * 1e6)


def finalize_filter(raw_int, notch_hz=50.0):
    from scipy.signal import sosfiltfilt, butter, iirnotch, filtfilt
    uv = raw_to_uv(raw_int[:N_CAPTURE])
    uv -= uv.mean()
    sos_hp = butter(2, 1.0,  btype='high', fs=FS, output='sos')
    sos_lp = butter(4, 40.0, btype='low',  fs=FS, output='sos')
    uv = sosfiltfilt(sos_hp, uv)
    uv = sosfiltfilt(sos_lp, uv)
    if notch_hz > 0:
        b_n, a_n = iirnotch(notch_hz, 30.0, FS)
        uv = filtfilt(b_n, a_n, uv)
    return uv[EDGE_TRIM: EDGE_TRIM + N_RECORD]


# ─────────────────────────────────────────────────────────────────────────────
# SPECTROGRAM → PIL IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def eeg_to_spectrogram_image(uv: np.ndarray,
                              target_px: int = IMG_SIZE) -> Image.Image:
    idx_start = int(T_START * SPEC_FS)
    idx_end   = int(T_END   * SPEC_FS)
    seg = uv[idx_start:idx_end]
    if len(seg) == 0:
        raise ValueError(
            f"Recorded buffer ({len(uv)} samples) is shorter than the "
            f"trained window [{idx_start}:{idx_end}] — check N_RECORD."
        )

    nperseg = min(256, len(seg) // 2)
    freqs, seg_times, sxx = scipy_spectrogram(
        seg,
        fs=SPEC_FS,
        nperseg=nperseg,
        noverlap=int(nperseg * 0.8),
    )

    dpi = 100
    fig_in = target_px / dpi
    fig = Figure(figsize=(fig_in, fig_in), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.pcolormesh(
        seg_times, freqs,
        10 * np.log10(sxx + 1e-10),
        shading="gouraud", cmap="jet",
    )
    ax.set_ylim(0, min(45, SPEC_FS / 2))
    ax.set_xlim(seg_times[0], seg_times[-1])

    buf = BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight", pad_inches=0, format="png")
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)   # low freq → bottom of image
    img = img.resize((target_px, target_px), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(checkpoint_path: str, num_classes: int, device):
    model = build_model(num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


@torch.no_grad()
def classify(model, img: Image.Image, device) -> tuple[str, np.ndarray]:
    tensor = INFER_TRANSFORM(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = CLASSES[int(probs.argmax())]
    return pred, probs


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class LiveClassifyGUI(tk.Tk):
    def __init__(self, model_path: str, notch_default: float = 50.0,
                 ble_address: str = BLE_ADDRESS):
        super().__init__()
        self.title("EEG Live Classifier + BLE Motor")
        self.configure(bg=C["bg"])
        self.minsize(1200, 700)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Loading {model_path}  on {self.device}")
        self.model  = load_model(model_path, len(CLASSES), self.device)
        print("[Model] Ready.")

        # BLE
        self.ble = BLEManager(ble_address, BLE_CHAR_UUID)
        self.ble.status_cb = self._on_ble_status

        self.ser              = None
        self.running          = False
        self.recording        = False
        self.sample_queue     = queue.Queue()
        self.live_buffer      = np.zeros(N_LIVE)
        self.record_buffer_raw: list[int] = []
        self._new_samples     = False
        self._notch_hz        = notch_default
        self._auto_send_ble   = True   # toggle via checkbox

        self.pipeline = LiveFilterPipeline(sr=FS, notch_hz=notch_default)

        self._build_ui()
        self._refresh_ports()
        self._tick()

    # ── UI BUILD ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=C["panel"], height=55)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="⬡ EEG LIVE CLASSIFIER",
                 font=("Courier", 13, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(side=tk.LEFT, padx=15)

        self.status_lbl = tk.Label(bar, text="● EEG: Disconnected",
                                   font=("Courier", 10),
                                   fg=C["danger"], bg=C["panel"])
        self.status_lbl.pack(side=tk.LEFT, padx=5)

        tk.Label(bar, text="PORT:", font=("Courier", 9),
                 fg=C["muted"], bg=C["panel"]).pack(side=tk.LEFT, padx=(20, 4))
        self.port_var = tk.StringVar()
        self.port_cb  = ttk.Combobox(bar, textvariable=self.port_var,
                                     width=12, state="readonly")
        self.port_cb.pack(side=tk.LEFT, pady=15)
        tk.Button(bar, text="⟳", font=("Courier", 9, "bold"),
                  fg=C["accent"], bg=C["panel"], bd=0, cursor="hand2",
                  command=self._refresh_ports).pack(side=tk.LEFT, padx=5)

        tk.Label(bar, text="NOTCH:", font=("Courier", 9),
                 fg=C["muted"], bg=C["panel"]).pack(side=tk.LEFT, padx=(15, 4))
        self.notch_var = tk.StringVar(value="50 Hz")
        nc = ttk.Combobox(bar, textvariable=self.notch_var,
                          values=["50 Hz", "60 Hz", "Off"], width=6, state="readonly")
        nc.pack(side=tk.LEFT)
        nc.bind("<<ComboboxSelected>>", self._on_notch_change)

        self.conn_btn = tk.Button(bar, text="CONNECT EEG",
                                  font=("Courier", 9, "bold"),
                                  fg=C["bg"], bg=C["accent"], bd=0,
                                  padx=12, cursor="hand2",
                                  command=self._toggle_connect)
        self.conn_btn.pack(side=tk.LEFT, padx=10, pady=12)

        # ── BLE section in top bar ────────────────────────────────────────────
        tk.Frame(bar, bg=C["border"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.ble_status_lbl = tk.Label(
            bar, text="● BLE: Disconnected",
            font=("Courier", 10), fg=C["danger"], bg=C["panel"])
        self.ble_status_lbl.pack(side=tk.LEFT, padx=5)

        self.ble_btn = tk.Button(bar, text="CONNECT BLE",
                                 font=("Courier", 9, "bold"),
                                 fg=C["bg"], bg=C["accent2"], bd=0,
                                 padx=12, cursor="hand2",
                                 command=self._toggle_ble)
        self.ble_btn.pack(side=tk.LEFT, padx=5, pady=12)

        # Device info
        self.device_lbl = tk.Label(
            bar, text=f"[{str(self.device).upper()}]",
            font=("Courier", 9), fg=C["muted"], bg=C["panel"])
        self.device_lbl.pack(side=tk.RIGHT, padx=15)

        # ── Left panel ───────────────────────────────────────────────────────
        lp = tk.Frame(self, bg=C["panel"], width=260)
        lp.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        lp.pack_propagate(False)

        tk.Label(lp, text="CLASSIFY CONTROL",
                 font=("Courier", 10, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(15, 5))

        self.rec_btn = tk.Button(lp, text="▶ RECORD & CLASSIFY",
                                 font=("Courier", 10, "bold"),
                                 fg=C["bg"], bg=C["ok"], bd=0, pady=10,
                                 cursor="hand2", command=self._start_recording)
        self.rec_btn.pack(fill=tk.X, padx=15, pady=5)

        self.timer_lbl = tk.Label(lp, text=f"0.0s / {RECORD_SEC:.1f}s",
                                  font=("Courier", 14, "bold"),
                                  fg=C["text"], bg=C["panel"])
        self.timer_lbl.pack(pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("G.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["ok"], thickness=8)
        self.pbar = ttk.Progressbar(lp, variable=self.progress_var,
                                    maximum=N_CAPTURE,
                                    style="G.Horizontal.TProgressbar")
        self.pbar.pack(fill=tk.X, padx=15, pady=5)

        # Auto-send toggle
        self._auto_send_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lp, text="Auto-send to motor",
                       variable=self._auto_send_var,
                       font=("Courier", 8),
                       fg=C["muted"], bg=C["panel"],
                       selectcolor=C["border"],
                       activebackground=C["panel"],
                       activeforeground=C["text"]).pack(anchor=tk.W, padx=15, pady=2)

        # ── Result panel ─────────────────────────────────────────────────────
        tk.Frame(lp, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=10)
        tk.Label(lp, text="PREDICTION",
                 font=("Courier", 10, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(0, 8))

        self.pred_lbl = tk.Label(lp, text="—",
                                 font=("Courier", 26, "bold"),
                                 fg=C["muted"], bg=C["panel"])
        self.pred_lbl.pack(pady=(0, 5))

        self.conf_lbl = tk.Label(lp, text="confidence: —",
                                 font=("Courier", 9),
                                 fg=C["muted"], bg=C["panel"])
        self.conf_lbl.pack()

        # BLE send status
        self.ble_action_lbl = tk.Label(lp, text="motor: —",
                                       font=("Courier", 9),
                                       fg=C["muted"], bg=C["panel"])
        self.ble_action_lbl.pack(pady=(2, 0))

        # Manual send button
        self.send_btn = tk.Button(lp, text="↑ SEND TO MOTOR",
                                  font=("Courier", 8, "bold"),
                                  fg=C["bg"], bg=C["muted"], bd=0, pady=5,
                                  cursor="hand2",
                                  command=self._manual_send,
                                  state=tk.DISABLED)
        self.send_btn.pack(fill=tk.X, padx=15, pady=6)
        self._last_pred = None

        # Confidence bars
        tk.Frame(lp, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=10)
        tk.Label(lp, text="CLASS PROBABILITIES",
                 font=("Courier", 9, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(0, 5))

        self._bar_vars = {}
        self._bar_labels = {}
        for cls in CLASSES:
            row = tk.Frame(lp, bg=C["panel"])
            row.pack(fill=tk.X, padx=15, pady=2)
            tk.Label(row, text=f"{cls:<6}", font=("Courier", 9),
                     fg=CLASS_COLORS.get(cls, C["text"]),
                     bg=C["panel"], width=6, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0)
            style.configure(f"{cls}.Horizontal.TProgressbar",
                            troughcolor=C["border"],
                            background=CLASS_COLORS.get(cls, C["accent"]),
                            thickness=12)
            pb = ttk.Progressbar(row, variable=var, maximum=1.0,
                                 style=f"{cls}.Horizontal.TProgressbar")
            pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
            pct = tk.Label(row, text="  0%", font=("Courier", 9),
                           fg=C["muted"], bg=C["panel"], width=5)
            pct.pack(side=tk.RIGHT)
            self._bar_vars[cls]   = var
            self._bar_labels[cls] = pct

        # Command reference
        tk.Frame(lp, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=8)
        tk.Label(lp, text="COMMAND MAP",
                 font=("Courier", 9, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15)
        cmd_map = [
            ("long",  "'1'", "▶▶ fwd 3 s"),
            ("short", "'2'", "▶  fwd 1 s"),
            ("back",  "'3'", "◀◀ bwd 3 s"),
            ("extra", "'4'", "◀  bwd 1 s"),
        ]
        for cls, cmd, desc in cmd_map:
            row = tk.Frame(lp, bg=C["panel"])
            row.pack(fill=tk.X, padx=15)
            tk.Label(row, text=f"{cls:<6}", font=("Courier", 8),
                     fg=CLASS_COLORS.get(cls, C["text"]),
                     bg=C["panel"], width=6, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=f"{cmd} {desc}",
                     font=("Courier", 8), fg=C["muted"],
                     bg=C["panel"]).pack(side=tk.LEFT)

        # ── Right: charts ─────────────────────────────────────────────────────
        rp = tk.Frame(self, bg=C["bg"])
        rp.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                padx=(5, 10), pady=10)

        self.fig = plt.figure(facecolor=C["bg"], figsize=(9, 6))
        gs  = self.fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                                    left=0.07, right=0.97,
                                    top=0.93, bottom=0.10)

        self.ax_live   = self.fig.add_subplot(gs[0, :])
        self.ax_full   = self.fig.add_subplot(gs[1, 0])
        self.ax_spec   = self.fig.add_subplot(gs[1, 1])

        _chart_style(self.ax_live,  "Live Waveform (Rolling 2 s)", "Time Context Window", "µV")
        _chart_style(self.ax_full,  "Recorded Waveform (4 s)",     "Time (s)",             "µV")
        _chart_style(self.ax_spec,  "Spectrogram fed to model",    "Time (s)",             "Hz")

        self.live_x    = np.linspace(-2, 0, N_LIVE)
        self.live_line, = self.ax_live.plot(
            self.live_x, self.live_buffer, color=C["accent"], lw=0.7)
        self.ax_live.set_xlim(-2, 0)
        self.ax_live.set_ylim(-50, 50)

        self.full_x    = np.linspace(0, RECORD_SEC, N_RECORD)
        self.full_line, = self.ax_full.plot(
            self.full_x, np.full(N_RECORD, np.nan),
            color=C["accent2"], lw=0.8)
        self.ax_full.set_xlim(0, RECORD_SEC)
        self.ax_full.set_ylim(-50, 50)

        self.canvas = FigureCanvasTkAgg(self.fig, master=rp)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── BLE CALLBACKS ─────────────────────────────────────────────────────────
    def _on_ble_status(self, msg: str, ok: bool):
        """Called from BLE thread → schedule UI update on main thread."""
        self.after(0, lambda: self._update_ble_ui(msg, ok))

    def _update_ble_ui(self, msg: str, ok: bool):
        color = C["ok"] if ok else C["danger"]
        self.ble_status_lbl.config(text=f"● {msg}", fg=color)
        if ok:
            self.ble_btn.config(text="DISCONNECT BLE", bg=C["danger"])
        else:
            self.ble_btn.config(text="CONNECT BLE", bg=C["accent2"])

    def _toggle_ble(self):
        if self.ble.connected:
            self.ble.disconnect()
        else:
            self.ble.connect()

    # ── HELPERS ──────────────────────────────────────────────────────────────
    def _on_notch_change(self, _=None):
        v = self.notch_var.get()
        hz = 50.0 if v == "50 Hz" else (60.0 if v == "60 Hz" else 0.0)
        self._notch_hz = hz
        self.pipeline.set_notch(hz)

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_cb["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.ser and self.ser.is_open:
            self.running = False
            try: self.ser.close()
            except: pass
            self.status_lbl.config(text="● EEG: Disconnected", fg=C["danger"])
            self.conn_btn.config(text="CONNECT EEG", bg=C["accent"])
        else:
            port = self.port_var.get()
            if not port:
                messagebox.showerror("Error", "No port selected.")
                return
            try:
                self.ser = serial.Serial(port, 230400, timeout=0.1)
                time.sleep(0.4)
                self.ser.write(b"c:1;\n")
                self.ser.flushInput()
                self.running     = True
                self.live_buffer = np.zeros(N_LIVE)
                self.pipeline.reset()
                self.status_lbl.config(text=f"● EEG: {port}", fg=C["ok"])
                self.conn_btn.config(text="DISCONNECT EEG", bg=C["danger"])
                threading.Thread(target=self._read_serial_worker,
                                 daemon=True).start()
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))

    def _read_serial_worker(self):
        buf = bytearray()
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    if chunk:
                        buf.extend(chunk)
                        parsed, buf = parse_byb_stream(buf)
                        if parsed:
                            self.sample_queue.put(parsed)
                else:
                    time.sleep(0.005)
            except Exception:
                break

    # ── RECORDING & PIPELINE ─────────────────────────────────────────────────
    def _start_recording(self):
        if not (self.ser and self.ser.is_open):
            messagebox.showwarning("Not Connected",
                                   "Connect to the EEG device first.")
            return
        if self.recording:
            return
        self.record_buffer_raw = []
        self.recording = True
        self.rec_btn.config(text="⏺ RECORDING...",
                            bg=C["warn"], state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.ble_action_lbl.config(text="motor: —", fg=C["muted"])
        self.full_line.set_ydata(np.full(N_RECORD, np.nan))
        self.ax_spec.cla()
        _chart_style(self.ax_spec, "Spectrogram fed to model", "Time (s)", "Hz")
        self.ax_spec.text(0.5, 0.5, "Recording...",
                          transform=self.ax_spec.transAxes,
                          color=C["muted"], ha="center", va="center",
                          fontfamily="monospace", fontsize=9)
        self.canvas.draw_idle()

    def _finalize_recording(self):
        self.recording = False

        uv = finalize_filter(self.record_buffer_raw, self._notch_hz)
        self.last_uv = uv

        self.full_line.set_ydata(uv)
        mn, mx = uv.min(), uv.max()
        margin = max((mx - mn) * 0.1, 5.0)
        self.ax_full.set_ylim(mn - margin, mx + margin)

        spec_img = eeg_to_spectrogram_image(uv, target_px=IMG_SIZE)
        self._show_spectrogram(spec_img)
        self.canvas.draw_idle()

        def _infer():
            pred, probs = classify(self.model, spec_img, self.device)
            self.after(0, lambda: self._show_result(pred, probs))

        threading.Thread(target=_infer, daemon=True).start()

    def _show_spectrogram(self, img: Image.Image):
        self.ax_spec.cla()
        _chart_style(self.ax_spec, "Spectrogram fed to model", "Time (s)", "Hz")
        arr = np.asarray(img)
        self.ax_spec.imshow(arr, aspect="auto",
                            extent=[0, RECORD_SEC, 0, 45], origin="lower")
        self.ax_spec.set_xlim(0, RECORD_SEC)
        self.ax_spec.set_ylim(0, 45)

    def _show_result(self, pred: str, probs: np.ndarray):
        color = CLASS_COLORS.get(pred, C["accent"])
        self.pred_lbl.config(text=pred.upper(), fg=color)
        conf = probs.max() * 100
        self.conf_lbl.config(text=f"confidence: {conf:.1f}%", fg=color)

        for i, cls in enumerate(CLASSES):
            p = float(probs[i])
            self._bar_vars[cls].set(p)
            self._bar_labels[cls].config(text=f"{p*100:4.1f}%")

        self._last_pred = pred

        # ── BLE send ──────────────────────────────────────────────────────────
        if self._auto_send_var.get():
            self._send_ble_command(pred)
        else:
            # Enable manual send button
            self.send_btn.config(state=tk.NORMAL, bg=C["accent2"])

        self.canvas.draw()
        self._reset_ui()

    def _send_ble_command(self, pred: str):
        cmd = CLASS_TO_CMD.get(pred)
        if cmd is None:
            return
        if not self.ble.connected:
            self.ble_action_lbl.config(
                text="motor: BLE not connected", fg=C["danger"])
            return
        self.ble.send_command(cmd)
        action_map = {
            "long":  "▶▶ fwd 3 s",
            "short": "▶  fwd 1 s",
            "back":  "◀◀ bwd 3 s",
            "extra": "◀  bwd 1 s",
        }
        desc = action_map.get(pred, "?")
        self.ble_action_lbl.config(
            text=f"motor: {pred} → {cmd.decode()} ({desc})",
            fg=CLASS_COLORS.get(pred, C["ok"]))

    def _manual_send(self):
        if self._last_pred:
            self._send_ble_command(self._last_pred)
            self.send_btn.config(state=tk.DISABLED, bg=C["muted"])

    def _reset_ui(self):
        self.rec_btn.config(text="▶ RECORD & CLASSIFY",
                            bg=C["ok"], state=tk.NORMAL)
        self.timer_lbl.config(text=f"0.0s / {RECORD_SEC:.1f}s", fg=C["text"])
        self.progress_var.set(0)

    # ── TICK (30 ms) ─────────────────────────────────────────────────────────
    def _tick(self):
        raw_chunks: list[int] = []
        try:
            while True:
                raw_chunks.extend(self.sample_queue.get_nowait())
        except queue.Empty:
            pass

        if raw_chunks:
            self._new_samples = True
            uv = self.pipeline.process(raw_chunks)
            n  = len(uv)
            if n >= N_LIVE:
                self.live_buffer = uv[-N_LIVE:]
            else:
                self.live_buffer = np.roll(self.live_buffer, -n)
                self.live_buffer[-n:] = uv

            if self.recording:
                self.record_buffer_raw.extend(raw_chunks)
                total = len(self.record_buffer_raw)
                self.progress_var.set(min(total, N_CAPTURE))
                display_sec = min(max(total - EDGE_TRIM, 0) / FS, RECORD_SEC)
                self.timer_lbl.config(
                    text=f"{display_sec:.1f}s / {RECORD_SEC:.1f}s",
                    fg=C["warn"])
                if total >= N_CAPTURE:
                    self._finalize_recording()

        if self._new_samples:
            self._new_samples = False
            self.live_line.set_ydata(self.live_buffer)
            mn, mx = self.live_buffer.min(), self.live_buffer.max()
            self.ax_live.set_ylim(mn - max((mx-mn)*0.1, 5),
                                  mx + max((mx-mn)*0.1, 5))
            self.canvas.draw_idle()

        self.after(30, self._tick)

    def close_app(self):
        self.running = False
        if self.ser:
            try: self.ser.close()
            except: pass
        self.ble.disconnect()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# TINY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _chart_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, color=C["muted"], fontsize=8,
                 fontfamily="monospace", loc="left", pad=4)
    ax.set_xlabel(xlabel, color=C["muted"], fontsize=7)
    ax.set_ylabel(ylabel, color=C["muted"], fontsize=7)
    ax.tick_params(colors=C["muted"], labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C["border"])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EEG Live Classifier + BLE Motor")
    parser.add_argument("--model",  default="eeg_classify.pth",
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--notch",  type=float, default=50.0,
                        help="Notch filter Hz (50, 60, or 0 to disable)")
    parser.add_argument("--port",   default=None,
                        help="Serial port for BYB SpikerShield (auto-detected if omitted)")
    parser.add_argument("--ble",    default=BLE_ADDRESS,
                        help=f"ESP32 BLE MAC address (default: {BLE_ADDRESS})")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("  Usage: python live_classify.py --model path/to/eeg_classify.pth")
        return

    app = LiveClassifyGUI(model_path=args.model,
                          notch_default=args.notch,
                          ble_address=args.ble)
    if args.port:
        app.port_var.set(args.port)
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()


if __name__ == "__main__":
    main()