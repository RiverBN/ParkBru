"""Camera-source selection and USB camera configuration."""

import os

import cv2
import numpy as np

def choose_camera_source_gui() -> tuple[str | None, str | None]:
    """Let the user choose a USB camera or enter an Android stream URL."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        choice = input("Camera source: [1] USB camera, [2] Android IP camera: ").strip()
        if choice == "2":
            url = input("Android video stream URL: ").strip()
            return ("android", url) if url else (None, None)
        return ("usb", None) if choice == "1" else (None, None)

    result: dict[str, str | None] = {"source": None, "url": None}
    root = tk.Tk()
    root.title("ParkBru camera source")
    root.resizable(False, False)
    root.geometry("560x265")
    body = ttk.Frame(root, padding=18)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Choose the camera for parking detection", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    choice = tk.StringVar(value="usb")
    ttk.Radiobutton(body, text="USB camera connected to this computer", variable=choice, value="usb").pack(anchor="w", pady=(14, 2))
    ttk.Radiobutton(body, text="Android IP camera over Wi-Fi", variable=choice, value="android").pack(anchor="w")
    ttk.Label(body, text="Android video stream URL (for example, https://PHONE_IP:4443/video/mjpeg)").pack(anchor="w", pady=(14, 4))
    url_var = tk.StringVar()
    url_entry = ttk.Entry(body, textvariable=url_var, width=72)
    url_entry.pack(fill="x")
    status = tk.StringVar(value="USB camera setup will open next.")
    ttk.Label(body, textvariable=status).pack(anchor="w", pady=(8, 0))

    def update_choice(*_args: object) -> None:
        enabled = choice.get() == "android"
        url_entry.configure(state="normal" if enabled else "disabled")
        status.set("Enter the camera app's video URL." if enabled else "USB camera setup will open next.")

    choice.trace_add("write", update_choice)
    update_choice()

    def accept() -> None:
        if choice.get() == "android":
            url = url_var.get().strip()
            if not url.startswith(("http://", "https://", "rtsp://", "rtsps://")):
                messagebox.showerror("Invalid stream URL", "Use an http://, https://, rtsp://, or rtsps:// URL.", parent=root)
                url_entry.focus_set()
                return
            result["url"] = url
        result["source"] = choice.get()
        root.destroy()

    buttons = ttk.Frame(body)
    buttons.pack(fill="x", pady=(16, 0))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(0, 8))
    ttk.Button(buttons, text="Continue", command=accept).pack(side="right")
    root.bind("<Return>", lambda _event: accept())
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result["source"], result["url"]

def open_camera(index: int) -> cv2.VideoCapture:
    """Open a local camera, preferring DirectShow on Windows."""
    if os.name == "nt":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(index)

def list_cameras(max_index: int = 10) -> list[int]:
    """Probe camera indices and return those that can provide a frame."""
    available: list[int] = []
    for index in range(max_index):
        capture = open_camera(index)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    available.append(index)
        finally:
            capture.release()
    return available

def configure_camera_gui(initial_index: int) -> tuple[cv2.VideoCapture | None, int]:
    """Show camera controls without a preview, then return the capture."""
    import tkinter as tk
    from tkinter import ttk

    cameras = list_cameras()
    if initial_index not in cameras and cameras:
        initial_index = cameras[0]
    elif initial_index not in cameras:
        cameras.append(initial_index)
    camera = open_camera(initial_index)
    if not camera.isOpened():
        return None, initial_index
    camera_defaults = {
        initial_index: (
            int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(camera.get(cv2.CAP_PROP_FOURCC)),
        )
    }

    root = tk.Tk()
    root.title("ParkBru camera setup")
    root.resizable(False, False)
    selected_camera = tk.IntVar(value=initial_index)
    autofocus = tk.BooleanVar(value=True)
    resolution = tk.StringVar(value="Camera default")
    frame_rate = tk.StringVar(value="Camera default")
    video_format = tk.StringVar(value="Auto (MJPEG for HD)")
    accepted = False
    status = tk.StringVar(value="Adjust settings, then click Use camera.")
    controls: dict[str, tuple[tk.Scale, tk.StringVar, int, int]] = {}

    body = ttk.Frame(root, padding=12)
    body.grid(row=0, column=0, sticky="nsew")
    settings = ttk.Frame(body)
    settings.grid(row=0, column=0, sticky="new")
    ttk.Label(settings, text="Camera").grid(row=0, column=0, sticky="w")
    camera_box = ttk.Combobox(
        settings,
        state="readonly",
        values=[f"Camera {index}" for index in cameras],
        width=24,
    )
    camera_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    camera_box.current(cameras.index(initial_index))
    ttk.Label(settings, text="Resolution").grid(row=2, column=0, sticky="w")
    resolution_box = ttk.Combobox(
        settings,
        state="readonly",
        textvariable=resolution,
        values=("Camera default", "640 × 480", "1280 × 720", "1920 × 1080"),
        width=24,
    )
    resolution_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(settings, text="Frame rate").grid(row=4, column=0, sticky="w")
    frame_rate_box = ttk.Combobox(
        settings,
        state="readonly",
        textvariable=frame_rate,
        values=("Camera default", "15 FPS", "24 FPS", "30 FPS", "60 FPS"),
        width=24,
    )
    frame_rate_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(settings, text="Video format").grid(row=6, column=0, sticky="w")
    format_box = ttk.Combobox(
        settings,
        state="readonly",
        textvariable=video_format,
        values=("Auto (MJPEG for HD)", "Camera default", "MJPEG (USB bandwidth saver)"),
        width=24,
    )
    format_box.grid(row=7, column=0, sticky="ew", pady=(0, 8))
    controls_area = ttk.Frame(settings)
    controls_area.grid(row=8, column=0, sticky="ew")

    property_specs = (
        ("Brightness", cv2.CAP_PROP_BRIGHTNESS, 0, 255),
        ("Contrast", cv2.CAP_PROP_CONTRAST, 0, 255),
        ("Saturation", cv2.CAP_PROP_SATURATION, 0, 255),
        ("Sharpness", cv2.CAP_PROP_SHARPNESS, 0, 255),
        ("Gain", cv2.CAP_PROP_GAIN, 0, 255),
        ("Exposure", cv2.CAP_PROP_EXPOSURE, -13, 0),
        ("Focus", cv2.CAP_PROP_FOCUS, 0, 255),
    )
    def rebuild_controls() -> None:
        for child in controls_area.winfo_children():
            child.destroy()
        controls.clear()
        for offset, (name, prop, minimum, maximum) in enumerate(property_specs, start=4):
            row = ttk.Frame(controls_area)
            row.grid(row=offset - 4, column=0, sticky="ew")
            ttk.Label(row, text=name, width=11).pack(side="left")
            info = tk.StringVar(value="Driver support unknown")
            slider = tk.Scale(
                row,
                from_=minimum,
                to=maximum,
                orient="horizontal",
                showvalue=True,
                length=180,
                resolution=1,
                command=lambda value, p=prop, label=info: apply_property(p, value, label),
            )
            slider.pack(side="left")
            ttk.Label(row, textvariable=info, width=23).pack(side="left", padx=(4, 0))
            controls[name] = (slider, info, prop, minimum)
            slider.set(minimum)
        autofocus_check = ttk.Checkbutton(
            controls_area,
            text="Autofocus",
            variable=autofocus,
            command=lambda: apply_autofocus(autofocus.get()),
        )
        autofocus_check.grid(row=len(property_specs), column=0, sticky="w", pady=(4, 0))
        refresh_controls()

    def apply_property(prop: int, value: str, label: tk.StringVar) -> None:
        try:
            requested = float(value)
            did_set = camera.set(prop, requested)
            actual = camera.get(prop)
            if did_set and abs(actual - requested) <= max(1.0, abs(requested) * 0.05):
                label.set(f"Active: {actual:g}")
            else:
                label.set("Not supported by camera")
        except cv2.error:
            label.set("Not supported by camera")

    def apply_autofocus(enabled: bool) -> None:
        supported = camera.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if enabled else 0.0)
        status.set("Autofocus updated." if supported else "Autofocus is not supported by this camera.")

    def refresh_controls() -> None:
        for slider, info, prop, minimum in controls.values():
            actual = camera.get(prop)
            if np.isfinite(actual):
                slider.set(min(max(round(actual), minimum), int(slider.cget("to"))))
            info.set(f"Current: {actual:g}" if actual != 0 else "Driver support unknown")

    def reopen_camera(index: int) -> None:
        nonlocal camera
        camera.release()
        camera = open_camera(index)
        if not camera.isOpened():
            status.set(f"Could not open camera {index}.")
            return
        camera_defaults.setdefault(
            index,
            (
                int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(camera.get(cv2.CAP_PROP_FOURCC)),
            ),
        )
        apply_resolution()
        rebuild_controls()
        apply_autofocus(autofocus.get())
        status.set(f"Previewing camera {index}.")

    def on_camera_change(_event: object = None) -> None:
        index = int(camera_box.get().split()[-1])
        selected_camera.set(index)
        reopen_camera(index)

    def apply_resolution(_event: object = None) -> None:
        choice = resolution.get()
        dimensions = {
            "640 × 480": (640, 480),
            "1280 × 720": (1280, 720),
            "1920 × 1080": (1920, 1080),
        }
        default_width, default_height, default_fourcc = camera_defaults[selected_camera.get()]
        width, height = dimensions.get(choice, (default_width, default_height))
        use_mjpeg = (
            video_format.get() == "MJPEG (USB bandwidth saver)"
            or (
                video_format.get() == "Auto (MJPEG for HD)"
                and (width >= 1920 or height >= 1080)
            )
        )
        if use_mjpeg:
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            camera.set(cv2.CAP_PROP_FOURCC, default_fourcc)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps_choice = frame_rate.get()
        if fps_choice != "Camera default":
            camera.set(cv2.CAP_PROP_FPS, int(fps_choice.split()[0]))
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((actual_fourcc >> (8 * shift)) & 0xFF) for shift in range(4)).strip()
        if not codec or not codec.isprintable():
            codec = "unknown format"
        status.set(f"Camera reports {actual_width} × {actual_height}, {actual_fps:g} FPS, {codec}.")

    camera_box.bind("<<ComboboxSelected>>", on_camera_change)
    resolution_box.bind("<<ComboboxSelected>>", apply_resolution)
    frame_rate_box.bind("<<ComboboxSelected>>", apply_resolution)
    format_box.bind("<<ComboboxSelected>>", apply_resolution)

    def finish() -> None:
        nonlocal accepted
        accepted = True
        root.destroy()

    def cancel() -> None:
        root.destroy()

    root.bind_all("<Return>", lambda _event: finish())
    root.bind_all("<Escape>", lambda _event: cancel())

    buttons = ttk.Frame(settings)
    buttons.grid(row=15, column=0, sticky="e", pady=(8, 0))
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right", padx=(6, 0))
    ttk.Button(buttons, text="Use camera", command=finish).pack(side="right")
    ttk.Label(body, textvariable=status).grid(row=1, column=0, sticky="w", pady=(4, 0))
    rebuild_controls()

    root.protocol("WM_DELETE_WINDOW", cancel)
    apply_autofocus(autofocus.get())
    root.mainloop()
    if accepted and camera.isOpened():
        return camera, selected_camera.get()
    camera.release()
    return None, selected_camera.get()

