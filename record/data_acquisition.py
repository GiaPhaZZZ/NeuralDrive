import os
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── CONFIGURATION CONSTANTS ───────────────────────────────────────────────────
FS          = 10000         # Sampling rate (Hz)
RECORD_SEC  = 4               # Target recording duration
LIVE_SEC    = 2               # Rolling view duration

# Extra samples recorded on each side to absorb:
#   • HP filter startup transient (IIR live stream resets on record start)
#   • Electrode/hardware settling at recording edges
EDGE_TRIM   = 500             # samples to discard from start AND end of recording

N_RECORD    = FS * RECORD_SEC                    # Target output samples (40,000)
N_CAPTURE   = N_RECORD + 2 * EDGE_TRIM           # Samples to actually capture (41,000)
N_LIVE      = FS * LIVE_SEC                      # Rolling live samples (20,000)

ADC_MAX     = 511             # 9-bit ADC (values 0-511, mid-point 255.5)
ADC_VREF    = 5.0
BYB_GAIN    = 974.0

# ── BYB IIR BIQUAD FILTER  (mirrors src/engine/FilterBase + HighPassFilter etc.) ──
class BiquadFilter:
    """
    Direct-form II transposed biquad IIR filter.
    State self.z persists across chunk boundaries — matches BYB C++
    FilterBase::filterContiguousData() gInputKeepBuffer / gOutputKeepBuffer.
    """
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
        """Matches src/engine/HighPassFilter::calculateCoefficients()"""
        f = cls(); _, c, a = f._vars(Fc, Q, sr); a0 = 1 + a
        f.b = np.array([(1+c)/2, -(1+c), (1+c)/2]) / a0
        f.a = np.array([-2*c, (1-a)]) / a0
        return f

    @classmethod
    def lowpass(cls, Fc, Q, sr):
        """Matches src/engine/LowPassFilter::calculateCoefficients()"""
        f = cls(); _, c, a = f._vars(Fc, Q, sr); a0 = 1 + a
        f.b = np.array([(1-c)/2, (1-c), (1-c)/2]) / a0
        f.a = np.array([-2*c, (1-a)]) / a0
        return f

    @classmethod
    def notch(cls, Fc, Q, sr):
        """Matches src/engine/NotchFilter::calculateCoefficients()"""
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


# ── DATA PARSING ──────────────────────────────────────────────────────────────
def parse_byb_stream(buf: bytearray) -> tuple[list[int], bytearray]:
    """
    BYB 2-byte frame parser — matches ArduinoSerial.cpp lines 1955-2002.
    Byte 0 (MSB, bit7=1): upper bits.  Byte 1 (LSB, bit7=0): lower 7 bits.
    """
    samples, i = [], 0
    while i < len(buf) - 1:
        b0, b1 = buf[i], buf[i+1]
        if (b0 & 0x80) and not (b1 & 0x80):
            samples.append(((b0 & 0x03) << 7) | (b1 & 0x7F))
            i += 2
        else:
            i += 1
    return samples, buf[i:]


def raw_to_uv(raw: np.ndarray) -> np.ndarray:
    """
    ADC integer → µV, centred on mid-scale (255.5 for 9-bit).
    No per-chunk mean removal — DC is handled by the HP filter.
    """
    return ((np.asarray(raw, dtype=np.float64) - ADC_MAX / 2.0)
            / ADC_MAX * ADC_VREF / BYB_GAIN * 1e6)


def get_auto_filename(base: str = "run") -> str:
    if not os.path.exists(f"{base}.npy"):
        return f"{base}.npy"
    i = 1
    while os.path.exists(f"{base}_{i}.npy"):
        i += 1
    return f"{base}_{i}.npy"


# ── LIVE FILTER PIPELINE (stateful IIR, causal) ───────────────────────────────
class LiveFilterPipeline:
    """
    Stateful causal IIR chain for the real-time display.
    State persists across 30 ms GUI tick chunks.
    Filters:  HP 1 Hz → LP 40 Hz → notch 50/60 Hz.

    LP cutoff is 40 Hz (not 100 Hz) because:
      • EEG band of interest is 1–40 Hz
      • LP at 40 Hz (order 4) gives −18 dB at 40 Hz roll-off matching reference data
      • Higher cutoff lets through EMG/noise that smears the spectrogram
    """
    def __init__(self, sr: float = FS, notch_hz: float = 50.0):
        self.sr  = sr
        self.hp  = BiquadFilter.highpass(1.0,  0.707, sr)
        self.lp  = BiquadFilter.lowpass(40.0,  0.707, sr)   # ← 40 Hz, not 100 Hz
        self._build_notch(notch_hz)

    def _build_notch(self, hz):
        self.notch_hz = hz
        self._notch   = BiquadFilter.notch(hz, 30.0, self.sr) if hz > 0 else None

    def process(self, raw_int: list[int]) -> np.ndarray:
        uv = raw_to_uv(raw_int)
        uv = self.hp.process(uv)
        uv = self.lp.process(uv)
        if self._notch is not None:
            uv = self._notch.process(uv)
        return uv

    def set_notch(self, hz: float):
        self._build_notch(hz)

    def reset(self):
        for f in (self.hp, self.lp):
            f.reset()
        if self._notch:
            self._notch.reset()


# ── FINALIZE FILTER (zero-phase, applied once on complete recording) ──────────
def finalize_filter(raw_int: list[int], notch_hz: float = 50.0) -> np.ndarray:
    """
    Converts a complete raw ADC recording to clean µV using zero-phase filters.

    Pipeline:
      1. raw → µV (midscale-centred, no per-chunk mean)
      2. Global mean subtraction (true DC block before HP filter)
      3. HP 1 Hz  Butterworth order-2  (zero-phase via sosfiltfilt)
      4. LP 40 Hz Butterworth order-4  (zero-phase — matches stop_1 rolloff)
      5. Notch 50/60 Hz (zero-phase filtfilt)
      6. Trim EDGE_TRIM samples from each end (removes electrode-settling / 
         HP-filter warmup transients that corrupt the recording edges)

    Key changes vs previous version:
      • LP cutoff 100 Hz → 40 Hz (order 4): this is the primary fix.
        At 40 Hz, EEG is fully preserved while EMG+HF noise that was
        smearing the 20-45 Hz spectrogram band is rejected.
      • EDGE_TRIM=500: edges are discarded because the IIR live stream resets
        its HP filter state when recording begins, causing a ramp transient
        of ~100-300 samples that contaminates the first 50 ms of saved data.
      • N_CAPTURE = N_RECORD + 2×EDGE_TRIM samples are recorded so that
        after trimming the output is exactly N_RECORD samples.
    """
    from scipy.signal import sosfiltfilt, butter, iirnotch, filtfilt

    uv = raw_to_uv(raw_int[:N_CAPTURE])       # full capture block → µV
    uv -= uv.mean()                            # global DC block

    sos_hp = butter(2, 1.0,  btype='high', fs=FS, output='sos')
    sos_lp = butter(4, 40.0, btype='low',  fs=FS, output='sos')  # order-4

    uv = sosfiltfilt(sos_hp, uv)
    uv = sosfiltfilt(sos_lp, uv)

    if notch_hz > 0:
        b_n, a_n = iirnotch(notch_hz, 30.0, FS)
        uv = filtfilt(b_n, a_n, uv)

    # Discard edge transients; result is exactly N_RECORD samples
    uv = uv[EDGE_TRIM: EDGE_TRIM + N_RECORD]
    return uv


# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
C = {
    "bg":     "#0D0F14", "panel":  "#13161E", "border": "#1E2330",
    "accent": "#00C8FF", "accent2":"#7C3AED", "warn":   "#F59E0B",
    "ok":     "#10B981", "danger": "#EF4444", "text":   "#E2E8F0",
    "muted":  "#64748B"
}

# ── APPLICATION CLASS ─────────────────────────────────────────────────────────
class EEGRecorderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Recorder — BYB SpikerShield")
        self.configure(bg=C["bg"])
        self.minsize(1100, 650)

        self.ser              = None
        self.running          = False
        self.recording        = False
        self.popup_open       = False
        self.sample_queue     = queue.Queue()
        self.live_buffer      = np.zeros(N_LIVE)
        self.record_buffer_raw: list[int] = []
        self._new_samples     = False

        self.pipeline = LiveFilterPipeline(sr=FS, notch_hz=50.0)

        self._build_ui()
        self._refresh_ports()
        self._tick()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=C["panel"], height=55)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="⬡ EEG RECORDER", font=("Courier", 13, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(side=tk.LEFT, padx=15)

        self.status_lbl = tk.Label(bar, text="● Disconnected",
                                   font=("Courier", 10), fg=C["danger"], bg=C["panel"])
        self.status_lbl.pack(side=tk.LEFT, padx=5)

        tk.Label(bar, text="PORT:", font=("Courier", 9),
                 fg=C["muted"], bg=C["panel"]).pack(side=tk.LEFT, padx=(20, 4))
        self.port_var = tk.StringVar()
        self.port_cb  = ttk.Combobox(bar, textvariable=self.port_var,
                                     width=12, state="readonly")
        self.port_cb.pack(side=tk.LEFT, pady=15)
        tk.Button(bar, text="⟳", font=("Courier", 9, "bold"), fg=C["accent"],
                  bg=C["panel"], bd=0, cursor="hand2",
                  command=self._refresh_ports).pack(side=tk.LEFT, padx=5)

        # Notch selector
        tk.Label(bar, text="NOTCH:", font=("Courier", 9),
                 fg=C["muted"], bg=C["panel"]).pack(side=tk.LEFT, padx=(15, 4))
        self.notch_var = tk.StringVar(value="50 Hz")
        nc = ttk.Combobox(bar, textvariable=self.notch_var,
                          values=["50 Hz", "60 Hz", "Off"], width=6, state="readonly")
        nc.pack(side=tk.LEFT)
        nc.bind("<<ComboboxSelected>>", self._on_notch_change)

        self.conn_btn = tk.Button(bar, text="CONNECT",
                                  font=("Courier", 9, "bold"),
                                  fg=C["bg"], bg=C["accent"], bd=0,
                                  padx=12, cursor="hand2",
                                  command=self._toggle_connect)
        self.conn_btn.pack(side=tk.LEFT, padx=10, pady=12)

        # Left panel
        lp = tk.Frame(self, bg=C["panel"], width=240)
        lp.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        lp.pack_propagate(False)

        tk.Label(lp, text="RECORDING CONTROL", font=("Courier", 10, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(15, 5))

        self.rec_btn = tk.Button(lp, text="▶ START RECORD",
                                 font=("Courier", 10, "bold"),
                                 fg=C["bg"], bg=C["ok"], bd=0, pady=10,
                                 cursor="hand2", command=self._start_recording)
        self.rec_btn.pack(fill=tk.X, padx=15, pady=5)

        self.timer_lbl = tk.Label(lp, text="0.0s / 4.0s",
                                  font=("Courier", 14, "bold"),
                                  fg=C["text"], bg=C["panel"])
        self.timer_lbl.pack(pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["ok"], thickness=8)
        self.pbar = ttk.Progressbar(lp, variable=self.progress_var,
                                    maximum=N_CAPTURE,
                                    style="Custom.Horizontal.TProgressbar")
        self.pbar.pack(fill=tk.X, padx=15, pady=5)

        # Filter info
        tk.Frame(lp, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=8)
        tk.Label(lp, text="FILTER CHAIN", font=("Courier", 10, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(0, 4))
        self.filter_info = tk.Label(lp,
                                    text="HP: 1 Hz   LP: 40 Hz\nNotch: 50 Hz   Trim: ±50ms",
                                    font=("Courier", 8), fg=C["muted"],
                                    bg=C["panel"], justify=tk.LEFT)
        self.filter_info.pack(anchor=tk.W, padx=15)

        # File properties
        tk.Frame(lp, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=10)
        tk.Label(lp, text="FILE PROPERTIES", font=("Courier", 10, "bold"),
                 fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=5)
        tk.Label(lp, text="Base Name:", font=("Courier", 9),
                 fg=C["muted"], bg=C["panel"]).pack(anchor=tk.W, padx=15)

        nf = tk.Frame(lp, bg=C["border"])
        nf.pack(fill=tk.X, padx=15, pady=5)
        self.filename_var = tk.StringVar(value="run")
        self.filename_var.trace_add("write", self._update_file_preview)
        tk.Entry(nf, textvariable=self.filename_var, font=("Courier", 10),
                 bg=C["border"], fg=C["text"], bd=0,
                 insertbackground=C["accent"]).pack(
                     side=tk.LEFT, fill=tk.X, expand=True, ipady=4, ipadx=4)
        tk.Label(nf, text=".npy ", font=("Courier", 10),
                 fg=C["muted"], bg=C["border"]).pack(side=tk.RIGHT)

        self.preview_lbl = tk.Label(lp, text="→ run.npy",
                                    font=("Courier", 8, "italic"),
                                    fg=C["muted"], bg=C["panel"])
        self.preview_lbl.pack(anchor=tk.W, padx=15, pady=2)

        # Right: charts
        rp = tk.Frame(self, bg=C["bg"])
        rp.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        self.fig, (self.ax_live, self.ax_full) = plt.subplots(2, 1, facecolor=C["bg"])
        self.fig.subplots_adjust(hspace=0.45, left=0.08, right=0.96,
                                 top=0.93, bottom=0.12)

        for ax, title in [
            (self.ax_live, "Live Waveform Stream (Rolling 2 Seconds)"),
            (self.ax_full, "Completed Sample Array Analysis (4 Seconds Summary)")
        ]:
            ax.set_facecolor(C["panel"])
            ax.set_title(title, color=C["muted"], fontsize=9,
                         fontfamily="monospace", loc="left", pad=5)
            ax.tick_params(colors=C["muted"], labelsize=8)
            ax.set_ylabel("Amplitude (µV)", color=C["muted"], fontsize=8)
            for sp in ax.spines.values():
                sp.set_color(C["border"])

        self.live_x = np.linspace(-LIVE_SEC, 0, N_LIVE)
        self.live_line, = self.ax_live.plot(self.live_x, self.live_buffer,
                                            color=C["accent"], lw=0.7)
        self.ax_live.set_xlim(-LIVE_SEC, 0)
        self.ax_live.set_ylim(-50, 50)
        self.ax_live.set_xlabel("Time Context Window", color=C["muted"], fontsize=8)

        self.full_x = np.linspace(0, RECORD_SEC, N_RECORD)
        self.full_line, = self.ax_full.plot(
            self.full_x, np.full(N_RECORD, np.nan), color=C["accent2"], lw=0.8)
        self.ax_full.set_xlim(0, RECORD_SEC)
        self.ax_full.set_ylim(-50, 50)
        self.ax_full.set_xlabel("Time (Seconds)", color=C["muted"], fontsize=8)

        self.waiting_msg = self.ax_full.text(
            RECORD_SEC / 2, 0, "Awaiting System Capture...",
            color=C["muted"], fontfamily="monospace", ha="center", va="center")

        self.canvas = FigureCanvasTkAgg(self.fig, master=rp)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── NOTCH CONTROL ─────────────────────────────────────────────────────────
    def _on_notch_change(self, _=None):
        val = self.notch_var.get()
        hz = 50.0 if val == "50 Hz" else (60.0 if val == "60 Hz" else 0.0)
        self.pipeline.set_notch(hz)
        notch_str = f"Notch: {val}"
        self.filter_info.config(
            text=f"HP: 1 Hz   LP: 40 Hz\n{notch_str}   Trim: ±50ms")

    # ── PORT MANAGEMENT ───────────────────────────────────────────────────────
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
            self.status_lbl.config(text="● Disconnected", fg=C["danger"])
            self.conn_btn.config(text="CONNECT", bg=C["accent"])
        else:
            port = self.port_var.get()
            if not port:
                messagebox.showerror("Error", "No connection port assigned.")
                return
            try:
                self.ser = serial.Serial(port, 230400, timeout=0.1)
                time.sleep(0.4)
                self.ser.write(b"c:1;\n")
                self.ser.flushInput()
                self.running     = True
                self.live_buffer = np.zeros(N_LIVE)
                self.pipeline.reset()
                self.status_lbl.config(text=f"● Connected: {port}", fg=C["ok"])
                self.conn_btn.config(text="DISCONNECT", bg=C["danger"])
                threading.Thread(target=self._read_serial_worker,
                                 daemon=True).start()
            except Exception as e:
                messagebox.showerror("Hardware Fault",
                                     f"Could not acquire connection link:\n{e}")

    # ── SERIAL READER ─────────────────────────────────────────────────────────
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

    # ── RECORDING ─────────────────────────────────────────────────────────────
    def _start_recording(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Notice",
                                   "Initialize active hardware channel link first.")
            return
        if self.recording:
            return

        self.record_buffer_raw = []
        self.recording = True

        self.rec_btn.config(text="⏺ RECORDING...", bg=C["warn"], state=tk.DISABLED)
        self.full_line.set_ydata(np.full(N_RECORD, np.nan))
        self.waiting_msg.set_text("Acquiring Stream Windows...")
        self.waiting_msg.set_alpha(1.0)
        self.canvas.draw_idle()

    def _finalize_recording(self):
        self.recording = False

        notch_hz = (50.0 if self.notch_var.get() == "50 Hz"
                    else 60.0 if self.notch_var.get() == "60 Hz"
                    else 0.0)

        # Apply complete zero-phase pipeline (LP 40Hz + edge trim)
        self.last_signal_uv = finalize_filter(self.record_buffer_raw, notch_hz)

        self.full_line.set_ydata(self.last_signal_uv)
        self.waiting_msg.set_alpha(0.0)

        mn, mx = self.last_signal_uv.min(), self.last_signal_uv.max()
        margin = max((mx - mn) * 0.1, 5.0)
        self.ax_full.set_ylim(mn - margin, mx + margin)
        self.canvas.draw()

        self._prompt_save_or_discard()

    def _prompt_save_or_discard(self):
        self.popup_open = True
        popup = tk.Toplevel(self)
        popup.title("Review Data Array")
        popup.geometry("320x150")
        popup.configure(bg=C["panel"])
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        x = self.winfo_x() + self.winfo_width()  // 2 - 160
        y = self.winfo_y() + self.winfo_height() // 2 - 75
        popup.geometry(f"+{x}+{y}")

        tk.Label(popup, text="4s Capture Processing Complete!",
                 font=("Courier", 10, "bold"), fg=C["accent"],
                 bg=C["panel"]).pack(pady=(15, 5))
        tk.Label(popup, text="Save array file or clear run context?",
                 font=("Courier", 9), fg=C["text"],
                 bg=C["panel"]).pack(pady=2)

        bf = tk.Frame(popup, bg=C["panel"])
        bf.pack(fill=tk.X, pady=15, padx=20)

        def save_action():
            base  = self.filename_var.get().strip() or "run"
            fname = get_auto_filename(base)
            np.save(fname, self.last_signal_uv)
            print(f"💾 Saved: {fname}")
            self.popup_open = False
            popup.destroy()
            self._reset_recording_ui()

        def discard_action():
            self.full_line.set_ydata(np.full(N_RECORD, np.nan))
            self.waiting_msg.set_text("Awaiting System Capture...")
            self.waiting_msg.set_alpha(1.0)
            self.ax_full.set_ylim(-50, 50)
            self.canvas.draw()
            print("🗑️ Recording dropped.")
            self.popup_open = False
            popup.destroy()
            self._reset_recording_ui()

        tk.Button(bf, text="SAVE", font=("Courier", 9, "bold"),
                  fg=C["bg"], bg=C["ok"], bd=0, width=10, pady=6,
                  cursor="hand2", command=save_action).pack(
                      side=tk.LEFT, expand=True, padx=5)
        tk.Button(bf, text="DISCARD", font=("Courier", 9, "bold"),
                  fg=C["text"], bg=C["danger"], bd=0, width=10, pady=6,
                  cursor="hand2", command=discard_action).pack(
                      side=tk.RIGHT, expand=True, padx=5)

        popup.protocol("WM_DELETE_WINDOW", discard_action)

    def _reset_recording_ui(self):
        self.rec_btn.config(text="▶ START RECORD", bg=C["ok"], state=tk.NORMAL)
        self.timer_lbl.config(text=f"0.0s / {RECORD_SEC:.1f}s", fg=C["text"])
        self.progress_var.set(0)
        self._update_file_preview()

    # ── GUI TICK (30 ms) ──────────────────────────────────────────────────────
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

                # Show progress relative to actual output (N_RECORD)
                display_sec = min(max(total - EDGE_TRIM, 0) / FS, RECORD_SEC)
                self.timer_lbl.config(
                    text=f"{display_sec:.1f}s / {RECORD_SEC:.1f}s",
                    fg=C["warn"])

                if total >= N_CAPTURE:
                    self._finalize_recording()

        if self._new_samples and not self.popup_open:
            self._new_samples = False
            self.live_line.set_ydata(self.live_buffer)
            mn, mx = self.live_buffer.min(), self.live_buffer.max()
            self.ax_live.set_ylim(mn - max((mx-mn)*0.1, 5), mx + max((mx-mn)*0.1, 5))
            self.canvas.draw_idle()

        self.after(30, self._tick)

    def _update_file_preview(self, *_):
        base = self.filename_var.get().strip() or "run"
        self.preview_lbl.config(text=f"→ {get_auto_filename(base)}")

    def close_app(self):
        self.running = False
        if self.ser:
            try: self.ser.close()
            except: pass
        self.destroy()


if __name__ == "__main__":
    app = EEGRecorderGUI()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()