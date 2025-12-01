import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from dataclasses import dataclass, field


@dataclass
class StateValues:
    cam1_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam1_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam2_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam2_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_offsets: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "cam1_pos": self.cam1_pos.copy(),
            "cam1_rot": self.cam1_rot.copy(),
            "cam2_pos": self.cam2_pos.copy(),
            "cam2_rot": self.cam2_rot.copy(),
            "joint_offsets": self.joint_offsets.copy(),
        }


class StateTuner:
    """Runs Tkinter GUI in a background thread, providing live parameter tuning."""
    def __init__(self, joint_names=None):
        if joint_names is None:
            joint_names = []

        self.joint_names = joint_names
        self.state = StateValues()
        for j in joint_names:
            self.state.joint_offsets[j] = 0.0

        self.thread = None
        self.root = None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def start(self):
        """Launch the GUI in a background thread (non-blocking)."""
        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def get_state(self):
        """Return a copy of the current tuning values."""
        return self.state.as_dict()

    # ---------------------------------------------------------
    # Internal GUI setup
    # ---------------------------------------------------------
    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title("State Tuner")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        notebook.add(self._make_camera_frame("Camera 1", "cam1"), text="Camera 1")
        notebook.add(self._make_camera_frame("Camera 2", "cam2"), text="Camera 2")
        notebook.add(self._make_joint_frame(), text="Joints")

        self.root.mainloop()

    # ---------------------------------------------------------
    # GUI Components
    # ---------------------------------------------------------
    def _make_camera_frame(self, title, prefix):
        frame = ttk.Frame(self.root, padding=10)

        # Position fields
        ttk.Label(frame, text="Position (x, y, z)").grid(row=0, column=0, sticky="w")
        for i in range(3):
            var = tk.DoubleVar()
            var.trace_add("write", lambda *_,
                                       idx=i,
                                       p=prefix,
                                       v=var: self._update_vec(p + "_pos", idx, v))
            ttk.Spinbox(frame, from_=-10, to=10, increment=0.001, textvariable=var, width=10)\
                .grid(row=0, column=i + 1, padx=3)

        # Rotation fields
        ttk.Label(frame, text="Rotation (roll, pitch, yaw)").grid(row=1, column=0, sticky="w")
        for i in range(3):
            var = tk.DoubleVar()
            var.trace_add("write", lambda *_,
                                       idx=i,
                                       p=prefix,
                                       v=var: self._update_vec(p + "_rot", idx, v))
            ttk.Spinbox(frame, from_=-3.14, to=3.14, increment=0.001, textvariable=var, width=10)\
                .grid(row=1, column=i + 1, padx=3)

        return frame

    def _make_joint_frame(self):
        frame = ttk.Frame(self.root, padding=10)

        for r, joint in enumerate(self.joint_names):
            ttk.Label(frame, text=joint).grid(row=r, column=0, sticky="w")

            var = tk.DoubleVar()
            var.trace_add("write", lambda *_,
                                       j=joint,
                                       v=var: self._update_joint(j, v))

            ttk.Spinbox(frame, from_=-3.14, to=3.14, increment=0.001, textvariable=var, width=10)\
                .grid(row=r, column=1, padx=3)

        return frame

    # ---------------------------------------------------------
    # Update handlers
    # ---------------------------------------------------------
    def _update_vec(self, attr_name, idx, var):
        vec = getattr(self.state, attr_name)
        vec[idx] = var.get()

    def _update_joint(self, joint, var):
        self.state.joint_offsets[joint] = var.get()


if __name__ == "__main__":
    tuner = StateTuner([
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ])

    tuner.start()   # <-- non-blocking

    while True:
        state = tuner.get_state()
        print(state)