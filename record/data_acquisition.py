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
FS          = 10_000          # Sampling rate (Hz)
RECORD_SEC  = 4               # Target recording duration
LIVE_SEC    = 2               # Rolling view duration
N_RECORD    = FS * RECORD_SEC # Total samples needed (40,000)
N_LIVE      = FS * LIVE_SEC   # Rolling samples (20,000)

ADC_MAX     = 511             # 9-bit ADC
ADC_VREF    = 5.0
BYB_GAIN    = 974.0

# ── DATA PARSING LOGIC ────────────────────────────────────────────────────────
def parse_byb_stream(buf: bytearray) -> tuple[list[int], bytearray]:
    samples = []
    i = 0
    while i < len(buf) - 1:
        b0, b1 = buf[i], buf[i + 1]
        if (b0 & 0x80) and not (b1 & 0x80):
            samples.append(((b0 & 0x03) << 7) | (b1 & 0x7F))
            i += 2
        else:
            i += 1
    return samples, buf[i:]

def adc_to_uv(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    uv = ((raw / ADC_MAX) * ADC_VREF / BYB_GAIN) * 1e6
    if len(uv) > 0:
        uv -= np.mean(uv)
    return uv

def get_auto_filename(base: str = "run") -> str:
    if not os.path.exists(f"{base}.npy"):
        return f"{base}.npy"
    counter = 1
    while os.path.exists(f"{base}_{counter}.npy"):
        counter += 1
    return f"{base}_{counter}.npy"

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
C = {
    "bg": "#0D0F14", "panel": "#13161E", "border": "#1E2330",
    "accent": "#00C8FF", "accent2": "#7C3AED", "warn": "#F59E0B",
    "ok": "#10B981", "danger": "#EF4444", "text": "#E2E8F0", "muted": "#64748B"
}

# ── APPLICATION CLASS ─────────────────────────────────────────────────────────
class EEGRecorderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Recorder — BYB SpikerShield")
        self.configure(bg=C["bg"])
        self.minsize(1100, 650)

        # Threading & Hardware Buffers
        self.ser = None
        self.running = False
        self.recording = False
        self.popup_open = False # Explicit state tracker for prompt windows
        self.sample_queue = queue.Queue()
        
        self.live_buffer = np.zeros(N_LIVE)
        self.record_buffer = []
        
        self._new_samples = False

        self._build_ui()
        self._refresh_ports()
        self._tick()  # Start the GUI update loop

    def _build_ui(self):
        # --- Top Action Bar ---
        bar = tk.Frame(self, bg=C["panel"], height=55)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="⬡ EEG RECORDER", font=("Courier", 13, "bold"), fg=C["accent"], bg=C["panel"]).pack(side=tk.LEFT, padx=15)
        
        self.status_lbl = tk.Label(bar, text="● Disconnected", font=("Courier", 10), fg=C["danger"], bg=C["panel"])
        self.status_lbl.pack(side=tk.LEFT, padx=5)

        # Port Selection Dropdown
        tk.Label(bar, text="PORT:", font=("Courier", 9), fg=C["muted"], bg=C["panel"]).pack(side=tk.LEFT, padx=(20, 4))
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(bar, textvariable=self.port_var, width=12, state="readonly")
        self.port_cb.pack(side=tk.LEFT, pady=15)
        
        tk.Button(bar, text="⟳", font=("Courier", 9, "bold"), fg=C["accent"], bg=C["panel"], bd=0, cursor="hand2", command=self._refresh_ports).pack(side=tk.LEFT, padx=5)
        
        self.conn_btn = tk.Button(bar, text="CONNECT", font=("Courier", 9, "bold"), fg=C["bg"], bg=C["accent"], bd=0, padx=12, cursor="hand2", command=self._toggle_connect)
        self.conn_btn.pack(side=tk.LEFT, padx=10, pady=12)

        # --- Left Side Controls ---
        left_panel = tk.Frame(self, bg=C["panel"], width=240)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        left_panel.pack_propagate(False)

        # Record Trigger Box
        tk.Label(left_panel, text="RECORDING CONTROL", font=("Courier", 10, "bold"), fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=(15, 5))
        
        self.rec_btn = tk.Button(left_panel, text="▶ START RECORD", font=("Courier", 10, "bold"), fg=C["bg"], bg=C["ok"], bd=0, pady=10, cursor="hand2", command=self._start_recording)
        self.rec_btn.pack(fill=tk.X, padx=15, pady=5)

        self.timer_lbl = tk.Label(left_panel, text="0.0s / 4.0s", font=("Courier", 14, "bold"), fg=C["text"], bg=C["panel"])
        self.timer_lbl.pack(pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", troughcolor=C["border"], background=C["ok"], thickness=8)
        self.pbar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=N_RECORD, style="Custom.Horizontal.TProgressbar")
        self.pbar.pack(fill=tk.X, padx=15, pady=5)

        # File Export Naming Layout
        tk.Frame(left_panel, bg=C["border"], height=1).pack(fill=tk.X, padx=10, pady=15)
        tk.Label(left_panel, text="FILE PROPERTIES", font=("Courier", 10, "bold"), fg=C["accent"], bg=C["panel"]).pack(anchor=tk.W, padx=15, pady=5)
        
        tk.Label(left_panel, text="Base Name:", font=("Courier", 9), fg=C["muted"], bg=C["panel"]).pack(anchor=tk.W, padx=15)
        
        name_entry_frame = tk.Frame(left_panel, bg=C["border"])
        name_entry_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.filename_var = tk.StringVar(value="run")
        self.filename_var.trace_add("write", self._update_file_preview)
        
        self.name_entry = tk.Entry(name_entry_frame, textvariable=self.filename_var, font=("Courier", 10), bg=C["border"], fg=C["text"], bd=0, insertbackground=C["accent"])
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, ipadx=4)
        tk.Label(name_entry_frame, text=".npy ", font=("Courier", 10), fg=C["muted"], bg=C["border"]).pack(side=tk.RIGHT)

        self.preview_lbl = tk.Label(left_panel, text="→ run.npy", font=("Courier", 8, "italic"), fg=C["muted"], bg=C["panel"])
        self.preview_lbl.pack(anchor=tk.W, padx=15, pady=2)

        # --- Right Side Charts Display ---
        right_panel = tk.Frame(self, bg=C["bg"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        self.fig, (self.ax_live, self.ax_full) = plt.subplots(2, 1, facecolor=C["bg"])
        self.fig.subplots_adjust(hspace=0.45, left=0.08, right=0.96, top=0.93, bottom=0.12)

        # Chart format properties
        for ax, title in [(self.ax_live, "Live Waveform Stream (Rolling 2 Seconds)"), 
                          (self.ax_full, "Completed Sample Array Analysis (4 Seconds Summary)")]:
            ax.set_facecolor(C["panel"])
            ax.set_title(title, color=C["muted"], fontsize=9, fontfamily="monospace", loc="left", pad=5)
            ax.tick_params(colors=C["muted"], labelsize=8)
            ax.set_ylabel("Amplitude (µV)", color=C["muted"], fontsize=8)
            for spine in ax.spines.values():
                spine.set_color(C["border"])

        # Live Line setup
        self.live_x = np.linspace(-LIVE_SEC, 0, N_LIVE)
        self.live_line, = self.ax_live.plot(self.live_x, self.live_buffer, color=C["accent"], lw=0.7)
        self.ax_live.set_xlim(-LIVE_SEC, 0)
        self.ax_live.set_ylim(-50, 50)
        self.ax_live.set_xlabel("Time Context Window", color=C["muted"], fontsize=8)

        # Full Sample Summary Setup
        self.full_x = np.linspace(0, RECORD_SEC, N_RECORD)
        self.full_line, = self.ax_full.plot(self.full_x, np.full(N_RECORD, np.nan), color=C["accent2"], lw=0.8)
        self.ax_full.set_xlim(0, RECORD_SEC)
        self.ax_full.set_ylim(-50, 50)
        self.ax_full.set_xlabel("Time (Seconds)", color=C["muted"], fontsize=8)
        
        self.waiting_msg = self.ax_full.text(RECORD_SEC / 2, 0, "Awaiting System Capture...", color=C["muted"], fontfamily="monospace", ha="center", va="center")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── HARDWARE PORT MANAGEMENT ──────────────────────────────────────────────────
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
                self.ser.write(b"c:1;\n") # Query streaming from shield
                self.ser.flushInput()
                
                self.running = True
                self.live_buffer = np.zeros(N_LIVE)
                self.status_lbl.config(text=f"● Connected: {port}", fg=C["ok"])
                self.conn_btn.config(text="DISCONNECT", bg=C["danger"])
                
                threading.Thread(target=self._read_serial_worker, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Hardware Fault", f"Could not acquire connection link:\n{e}")

    # ── BACKGROUND SERIAL DATA INGESTION WORKER ───────────────────────────────────
    def _read_serial_worker(self):
        local_buffer = bytearray()
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    if chunk:
                        local_buffer.extend(chunk)
                        parsed_samples, local_buffer = parse_byb_stream(local_buffer)
                        if parsed_samples:
                            self.sample_queue.put(parsed_samples)
                else:
                    time.sleep(0.005)
            except Exception:
                break

    # ── RECORDING TRACKERS ────────────────────────────────────────────────────────
    def _start_recording(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning("Notice", "Initialize active hardware channel link first.")
            return
        if self.recording:
            return

        self.record_buffer = []
        self.recording = True
        
        self.rec_btn.config(text="⏺ RECORDING...", bg=C["warn"], state=tk.DISABLED)
        self.full_line.set_ydata(np.full(N_RECORD, np.nan))
        self.waiting_msg.set_text("Acquiring Stream Windows...")
        self.waiting_msg.set_alpha(1.0)
        self.canvas.draw_idle()

    def _finalize_recording(self):
        self.recording = False
        
        # Keep exactly the target frame dimension (40,000 samples)
        raw_segment = np.array(self.record_buffer[:N_RECORD])
        self.last_signal_uv = adc_to_uv(raw_segment)
        
        # Plot the snapshot summary immediately
        self.full_line.set_ydata(self.last_signal_uv)
        self.waiting_msg.set_alpha(0.0) 
        
        mn, mx = np.min(self.last_signal_uv), np.max(self.last_signal_uv)
        margin = max((mx - mn) * 0.1, 5.0)
        self.ax_full.set_ylim(mn - margin, mx + margin)
        self.canvas.draw()

        # Fire pop-up dialogue prompt 
        self._prompt_save_or_discard()

    def _prompt_save_or_discard(self):
        """Creates a custom modal pop-up prompt window allowing data verification."""
        self.popup_open = True
        popup = tk.Toplevel(self)
        popup.title("Review Data Array")
        popup.geometry("320x150")
        popup.configure(bg=C["panel"])
        popup.resizable(False, False)
        popup.transient(self)  # Keep on top of main window
        popup.grab_set()       # Block interactions with main window until closed

        # Center the popup window relative to main window
        x = self.winfo_x() + (self.winfo_width() // 2) - 160
        y = self.winfo_y() + (self.winfo_height() // 2) - 75
        popup.geometry(f"+{x}+{y}")

        tk.Label(popup, text="4s Capture Processing Complete!", font=("Courier", 10, "bold"), fg=C["accent"], bg=C["panel"]).pack(pady=(15, 5))
        tk.Label(popup, text="Save array file or clear run context?", font=("Courier", 9), fg=C["text"], bg=C["panel"]).pack(pady=2)

        btn_frame = tk.Frame(popup, bg=C["panel"])
        btn_frame.pack(fill=tk.X, pady=15, padx=20)

        def save_action():
            base = self.filename_var.get().strip() or "run"
            saved_name = get_auto_filename(base)
            np.save(saved_name, self.last_signal_uv)
            print(f"💾 Captured sample saved to: {saved_name}")
            self.popup_open = False
            popup.destroy()
            self._reset_recording_ui()

        def discard_action():
            # Reset full graph back to empty state
            self.full_line.set_ydata(np.full(N_RECORD, np.nan))
            self.waiting_msg.set_text("Awaiting System Capture...")
            self.waiting_msg.set_alpha(1.0)
            self.ax_full.set_ylim(-50, 50)
            self.canvas.draw()
            print("🗑️ Recording sample dropped.")
            self.popup_open = False
            popup.destroy()
            self._reset_recording_ui()

        # Custom themed buttons
        save_btn = tk.Button(btn_frame, text="SAVE", font=("Courier", 9, "bold"), fg=C["bg"], bg=C["ok"], bd=0, width=10, pady=6, cursor="hand2", command=save_action)
        save_btn.pack(side=tk.LEFT, expand=True, padx=5)

        discard_btn = tk.Button(btn_frame, text="DISCARD", font=("Courier", 9, "bold"), fg=C["text"], bg=C["danger"], bd=0, width=10, pady=6, cursor="hand2", command=discard_action)
        discard_btn.pack(side=tk.RIGHT, expand=True, padx=5)

        # If user closes window via "X", default to discarding to safely reset UI
        popup.protocol("WM_DELETE_WINDOW", discard_action)

    def _reset_recording_ui(self):
        """Resets the left-panel progress gauges to zero for the next capture run."""
        self.rec_btn.config(text="▶ START RECORD", bg=C["ok"], state=tk.NORMAL)
        self.timer_lbl.config(text=f"0.0s / {RECORD_SEC:.1f}s", fg=C["text"])
        self.progress_var.set(0)
        self._update_file_preview()

    # ── MAIN SYSTEM TICK (RUNS ON GUI THREAD EVERY 30MS) ──────────────────────────
    def _tick(self):
        incoming_chunks = []
        try:
            while True:
                incoming_chunks.extend(self.sample_queue.get_nowait())
        except queue.Empty:
            pass

        if incoming_chunks:
            self._new_samples = True
            incoming_uv = adc_to_uv(incoming_chunks)
            n_samples = len(incoming_uv)

            # Always update live rolling buffer seamlessly
            if n_samples >= N_LIVE:
                self.live_buffer = incoming_uv[-N_LIVE:]
            else:
                self.live_buffer = np.roll(self.live_buffer, -n_samples)
                self.live_buffer[-n_samples:] = incoming_uv

            # Feed the recording storage sequence based on raw values
            if self.recording:
                self.record_buffer.extend(incoming_chunks)
                
                total_recorded = len(self.record_buffer)
                self.progress_var.set(min(total_recorded, N_RECORD))
                
                current_sec = min(total_recorded / FS, RECORD_SEC)
                self.timer_lbl.config(text=f"{current_sec:.1f}s / {RECORD_SEC:.1f}s", fg=C["warn"])
                
                if total_recorded >= N_RECORD:
                    self._finalize_recording()

        # Update Live Plot Stream smoothly (even while recording!)
        if self._new_samples and not self.popup_open:
            self._new_samples = False
            self.live_line.set_ydata(self.live_buffer)
            
            mn, mx = self.live_buffer.min(), self.live_buffer.max()
            margin = max((mx - mn) * 0.1, 5.0)
            self.ax_live.set_ylim(mn - margin, mx + margin)
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