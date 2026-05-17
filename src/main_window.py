from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not available, use system env vars

log = logging.getLogger(__name__)

# Load environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_ID", "my-first-project-igu3k")
ROBOFLOW_PROJECT_VERSION = os.getenv("ROBOFLOW_PROJECT_VERSION", "1")
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", r"E:\koding\Javascript\labeling-daun-itoh"))
ROBOFLOW_RESULTS_DIR = Path(os.getenv("ROBOFLOW_RESULTS_DIR", "roboflow/results"))
ROBOFLOW_TIMEOUT = int(os.getenv("ROBOFLOW_TIMEOUT", "180"))
POLL_INTERVAL = 0.5  # Check every 0.5 seconds

# Roboflow paths
ROBOFLOW_ROOT = PROJECT_ROOT / "roboflow"
ROBOFLOW_BATCH_FILE = ROBOFLOW_ROOT / "run_single.bat"

import cv2
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.annotation_engine import AnnotationCanvas, CanvasMode
from src.config import Config
from src.exporter import Exporter
from src.image_manager import ImageEntry, ImageManager, Polygon
from src.visualizer import Visualizer

try:
    from src.auto_labeler import AutoLabeler, AutoLabelWorker

    _HAS_SAM = True
except ImportError:
    _HAS_SAM = False
    AutoLabeler = None  # type: ignore[assignment,misc]
    AutoLabelWorker = None  # type: ignore[assignment,misc]


class _ThreadBridge(QObject):
    """Carries results from a plain Python thread back to the Qt main thread.

    Created on the main thread; when .result or .error is emitted from a
    background thread, Qt queues the signal and delivers it on the main thread's
    event loop — no QTimer.singleShot needed.
    """

    result = pyqtSignal(list)  # list[Polygon]
    error = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Roboflow integration helpers
# ---------------------------------------------------------------------------


def _polygon_color_from_class(class_name: str, instance_id: int) -> list[int]:
    """Generate consistent color based on class name and instance ID."""
    import hashlib

    hash_str = f"{class_name}_{instance_id}"
    hash_int = int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)
    return [(hash_int >> 16) & 0xFF, (hash_int >> 8) & 0xFF, hash_int & 0xFF]


def roboflow_to_polygons(
    roboflow_json: dict, instance_id_start: int = 1
) -> list[Polygon]:
    """Convert Roboflow JSON response to list of Polygon objects."""
    polygons = []
    predictions = roboflow_json.get("predictions", [])

    for idx, pred in enumerate(predictions, start=instance_id_start):
        # Convert points from {x, y} dict to [x, y] list
        points = [[p.get("x", 0), p.get("y", 0)] for p in pred.get("points", [])]

        if not points:
            continue

        class_name = pred.get("class", "unknown")
        class_id = pred.get("class_id", 0)
        confidence = pred.get("confidence", 0.0)

        polygons.append(
            Polygon(
                instance_id=idx,
                class_id=class_id,
                class_name=class_name,
                points=points,
                color=_polygon_color_from_class(class_name, idx),
                source="roboflow",
                confidence=confidence,
                occlusion_level=None,
                occlusion_ratio=None,
                occlusion_manual=False,
                is_confirmed=False,
            )
        )

    return polygons


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._img_manager = ImageManager(config)
        self._canvas = AnnotationCanvas(config, self)
        self._exporter = Exporter(config)
        self._visualizer = Visualizer(config)
        self._auto_labeler: AutoLabeler | None = None
        self._auto_worker = None
        self._current_entry: ImageEntry | None = None
        self._confidence: float = getattr(
            getattr(config, "sam", None), "auto_confidence_threshold", 0.75
        )
        # True once load_model() has finished (success or fail)
        self._model_load_attempted: bool = not _HAS_SAM
        # State for tracking unsaved changes
        self._has_unsaved_changes: bool = False
        self._initial_polygons_snapshot: list[Polygon] = []  # Snapshot when entering manual mode

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

        # Load MobileSAM for Select feature (point-based segmentation)
        if _HAS_SAM:
            self._auto_labeler = AutoLabeler(config)
            log.info("MainWindow: memulai load model MobileSAM untuk Select feature...")
            QTimer.singleShot(0, self._load_model_async)

        self._update_sam_controls()
        self._update_status_bar()

    # ------------------------------------------------------------------ #
    # UI setup                                                             #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        self.setWindowTitle(self._config.project_name)
        canvas_cfg = getattr(self._config, "canvas", None)
        self.resize(
            getattr(canvas_cfg, "default_width", 1200),
            getattr(canvas_cfg, "default_height", 800),
        )
        self._build_toolbar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._sb_file = QLabel()
        self._sb_mode = QLabel()
        self._sb_zoom = QLabel()
        sb = QStatusBar()
        sb.addWidget(self._sb_file, 2)
        sb.addWidget(self._sb_mode, 1)
        sb.addPermanentWidget(self._sb_zoom)
        self.setStatusBar(sb)
        self._status_bar_obj = sb

    def _build_toolbar(self) -> None:
        from PyQt5.QtWidgets import QToolButton, QMenu

        tb1 = self.addToolBar("Main")
        tb1.setObjectName("tb_main")

        self._act_open = QAction("Open Folder", self)
        self._act_save = QAction("Save", self)
        tb1.addAction(self._act_open)
        tb1.addAction(self._act_save)
        tb1.addSeparator()

        self._act_prev = QAction("◀ Prev", self)
        self._act_next = QAction("Next ▶", self)
        tb1.addAction(self._act_prev)
        tb1.addAction(self._act_next)
        tb1.addSeparator()

        self._act_auto_label = QAction("✨ Auto Label", self)
        self._act_auto_all = QAction("⚡ Auto All", self)
        tb1.addAction(self._act_auto_label)
        tb1.addAction(self._act_auto_all)
        tb1.addSeparator()

        # Export dropdown button
        self._act_export_menu = QAction("Export", self)
        export_btn = QToolButton()
        export_btn.setText("Export ▼")
        export_btn.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self)

        self._act_exp_coco = QAction("Export COCO Splits", self)
        self._act_exp_voc = QAction("Export XML", self)
        self._act_exp_vis = QAction("Export Visualized", self)
        export_menu.addAction(self._act_exp_coco)
        export_menu.addAction(self._act_exp_voc)
        export_menu.addAction(self._act_exp_vis)
        export_btn.setMenu(export_menu)
        tb1.addWidget(export_btn)
        tb1.addSeparator()

        self._act_zoom_in = QAction("+", self)
        self._act_zoom_out = QAction("-", self)
        self._act_zoom_fit = QAction("Fit", self)
        tb1.addAction(self._act_zoom_in)
        tb1.addAction(self._act_zoom_out)
        tb1.addAction(self._act_zoom_fit)

        self._act_open.triggered.connect(self.open_folder)
        self._act_save.triggered.connect(self.save_current)
        self._act_prev.triggered.connect(self.prev_image)
        self._act_next.triggered.connect(self.next_image)
        self._act_auto_label.triggered.connect(self.trigger_sam_auto_current)
        self._act_auto_all.triggered.connect(self.trigger_sam_auto_all)
        self._act_exp_coco.triggered.connect(self.export_coco_splits)
        self._act_exp_voc.triggered.connect(self.export_voc)
        self._act_exp_vis.triggered.connect(self.export_visualized)
        self._act_zoom_in.triggered.connect(self._canvas.zoom_in)
        self._act_zoom_out.triggered.connect(self._canvas.zoom_out)
        self._act_zoom_fit.triggered.connect(self._canvas.reset_zoom)

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Dataset stats
        stats_box = QGroupBox("Dataset")
        sl = QVBoxLayout(stats_box)
        self._lbl_dataset = QLabel("📁 Belum ada gambar")
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v/%m anotasi")
        self._lbl_auto = QLabel("🤖 0 auto-labeled")
        sl.addWidget(self._lbl_dataset)
        sl.addWidget(self._progress_bar)
        sl.addWidget(self._lbl_auto)
        layout.addWidget(stats_box)

        # Manual Label Control
        manual_group = QGroupBox("Manual Label Control")
        manual_l = QVBoxLayout(manual_group)

        # First row: Draw/Select buttons (visible when NOT in manual mode)
        self._btn_manual_draw = QPushButton("Draw")
        self._btn_manual_select = QPushButton("Select")
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._btn_manual_draw)
        mode_row.addWidget(self._btn_manual_select)
        manual_l.addLayout(mode_row)

        # Second row: Save/Cancel buttons (visible when in manual mode)
        self._btn_manual_save = QPushButton("Save")
        self._btn_manual_cancel = QPushButton("Cancel")
        self._btn_manual_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self._btn_manual_cancel.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        save_cancel_row = QHBoxLayout()
        save_cancel_row.addWidget(self._btn_manual_save)
        save_cancel_row.addWidget(self._btn_manual_cancel)
        manual_l.addLayout(save_cancel_row)

        # Third row: Undo/Redo buttons (always visible but enabled when has changes)
        self._btn_undo = QPushButton("Undo")
        self._btn_redo = QPushButton("Redo")
        self._btn_undo.setEnabled(False)
        self._btn_redo.setEnabled(False)
        undo_redo_row = QHBoxLayout()
        undo_redo_row.addWidget(self._btn_undo)
        undo_redo_row.addWidget(self._btn_redo)
        manual_l.addLayout(undo_redo_row)

        layout.addWidget(manual_group)

        # Auto Label Controls
        self._sam_group = QGroupBox("Auto Label Controls")
        sam_l = QVBoxLayout(self._sam_group)

        mode_row = QHBoxLayout()
        self._btn_point = QPushButton("Point")
        self._btn_box = QPushButton("Box")
        self._btn_auto_cur = QPushButton("Auto")
        mode_row.addWidget(self._btn_point)
        mode_row.addWidget(self._btn_box)
        mode_row.addWidget(self._btn_auto_cur)
        sam_l.addLayout(mode_row)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(50, 99)
        self._conf_slider.setValue(int(self._confidence * 100))
        self._conf_lbl = QLabel(f"{self._confidence:.2f}")
        self._conf_lbl.setMinimumWidth(32)
        conf_row.addWidget(self._conf_slider)
        conf_row.addWidget(self._conf_lbl)
        sam_l.addLayout(conf_row)

        ar_row = QHBoxLayout()
        self._btn_accept = QPushButton("✓ Accept All")
        self._btn_reject = QPushButton("✗ Reject All")
        ar_row.addWidget(self._btn_accept)
        ar_row.addWidget(self._btn_reject)
        sam_l.addLayout(ar_row)
        layout.addWidget(self._sam_group)

        # Instance list
        inst_box = QGroupBox("Instance List")
        il = QVBoxLayout(inst_box)
        self._inst_list = QListWidget()
        self._inst_list.setMaximumHeight(220)
        self._overflow_lbl = QLabel()
        self._overflow_lbl.setStyleSheet("color: gray; font-style: italic;")
        self._overflow_lbl.hide()
        il.addWidget(self._inst_list)
        il.addWidget(self._overflow_lbl)
        layout.addWidget(inst_box)

        # Thumbnail
        thumb_box = QGroupBox("Preview")
        tl = QVBoxLayout(thumb_box)
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setMinimumHeight(100)
        tl.addWidget(self._thumb)
        layout.addWidget(thumb_box)

        layout.addStretch()

        # Manual Label Control connections
        self._btn_manual_draw.clicked.connect(self.trigger_manual_draw)
        self._btn_manual_select.clicked.connect(self.trigger_manual_select)
        self._btn_manual_save.clicked.connect(self.finish_manual_save)
        self._btn_manual_cancel.clicked.connect(self.cancel_manual_select)
        self._btn_undo.clicked.connect(self.undo_action)
        self._btn_redo.clicked.connect(self.redo_action)

        # Auto Label Control connections
        self._btn_point.clicked.connect(self.trigger_sam_point)
        self._btn_box.clicked.connect(self.trigger_sam_box)
        self._btn_auto_cur.clicked.connect(self.trigger_sam_auto_current)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        self._btn_accept.clicked.connect(self.accept_all_auto)
        self._btn_reject.clicked.connect(self.reject_selected_auto)
        self._inst_list.itemClicked.connect(self._on_instance_clicked)

        return panel

    def _setup_shortcuts(self) -> None:
        def sc(key, slot):
            QShortcut(QKeySequence(key), self).activated.connect(slot)

        sc("N", self.next_image)
        sc("P", self.prev_image)
        sc("S", self.save_current)
        sc("D", self.trigger_manual_draw)
        sc("V", self.trigger_manual_select)
        sc("Return", self._handle_return_key)
        sc("Escape", self._handle_escape)
        sc("Delete", self._canvas.delete_selected)
        sc("Ctrl+Z", self._canvas.undo)
        sc("+", self._canvas.zoom_in)
        sc("-", self._canvas.zoom_out)
        sc("F", self._canvas.reset_zoom)

    def _connect_signals(self) -> None:
        self._canvas.annotation_changed.connect(self._on_annotation_changed)
        self._canvas.point_clicked.connect(self._on_point_clicked)
        self._canvas.point_refine_clicked.connect(self._on_point_refine_clicked)
        self._canvas.box_selected.connect(self._on_box_selected)
        self._canvas.has_unsaved_changes.connect(self._on_has_unsaved_changes)

    # ------------------------------------------------------------------ #
    # File & navigation                                                    #
    # ------------------------------------------------------------------ #

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Buka Folder Gambar")
        if not path:
            return
        count = self._img_manager.load_folder(path)
        log.info("open_folder: %d gambar ditemukan di %s", count, path)
        if count == 0:
            self._flash("Tidak ada gambar ditemukan")
            return
        self._load_image_to_canvas(self._img_manager.current_image())

    def save_current(self) -> None:
        if self._current_entry is None:
            return
        polygons = self._canvas.get_polygons()
        self._img_manager.save_annotation(self._current_entry, polygons)
        log.info(
            "save_current: %d polygon disimpan untuk %s",
            len(polygons),
            self._current_entry.filename,
        )
        self._flash("Tersimpan ✓")
        self._update_stats()

    def next_image(self) -> None:
        images = self._img_manager.get_all()
        if not images:
            return
        if self._img_manager.current_image().index >= len(images) - 1:
            self._flash("Gambar terakhir")
            return
        self._autosave_current()
        entry = self._img_manager.next()
        log.info(
            "next_image: navigasi ke %s (%d/%d)",
            entry.filename,
            entry.index + 1,
            len(images),
        )
        self._load_image_to_canvas(entry)

    def prev_image(self) -> None:
        images = self._img_manager.get_all()
        if not images:
            return
        if self._img_manager.current_image().index <= 0:
            self._flash("Gambar pertama")
            return
        self._autosave_current()
        entry = self._img_manager.prev()
        log.info(
            "prev_image: navigasi ke %s (%d/%d)",
            entry.filename,
            entry.index + 1,
            len(images),
        )
        self._load_image_to_canvas(entry)

    def _autosave_current(self) -> None:
        if self._current_entry is None:
            return
        polygons = self._canvas.get_polygons()
        self._img_manager.save_annotation(self._current_entry, polygons)
        self._flash("Tersimpan ✓")
        self._update_stats()

    # ------------------------------------------------------------------ #
    # SAM auto-labeling                                                    #
    # ------------------------------------------------------------------ #

    def trigger_sam_point(self) -> None:
        if not self._sam_ready():
            self._notify_sam_not_ready()
            return
        if self._current_entry is None:
            QMessageBox.information(
                self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu."
            )
            return
        log.info(
            "trigger_sam_point: mode SAM_POINT aktif, klik pada gambar untuk prediksi"
        )
        self._set_mode(CanvasMode.SAM_POINT)
        self._flash("Mode Point: Klik area kosong untuk buat baru, klik polygon untuk refine")

    def trigger_sam_box(self) -> None:
        if not self._sam_ready():
            self._notify_sam_not_ready()
            return
        if self._current_entry is None:
            QMessageBox.information(
                self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu."
            )
            return
        log.info(
            "trigger_sam_box: mode SAM_BOX aktif, seret kotak pada gambar untuk prediksi"
        )
        self._set_mode(CanvasMode.SAM_BOX)

    def trigger_sam_auto_current(self) -> None:
        if self._current_entry is None:
            QMessageBox.information(
                self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu."
            )
            return
        log.info(
            "trigger_sam_auto_current: mulai auto-label Roboflow untuk %s",
            self._current_entry.filename,
        )
        self._canvas.set_loading(True)
        # Show loading message in status bar
        self._flash("Roboflow: Memproses gambar...", ms=60000)

        image_path = self._current_entry.filepath
        image_name = Path(image_path).stem
        result_json_path = ROBOFLOW_RESULTS_DIR / f"{image_name}.json"

        # Clean up old result if exists
        if result_json_path.exists():
            result_json_path.unlink()

        bridge = _ThreadBridge(self)

        def _on_result(polygons: list) -> None:
            log.info(
                "trigger_sam_auto_current: UI thread menerima %d polygon, inject...",
                len(polygons),
            )
            self._canvas.set_loading(False)
            self._on_sam_result(polygons, source="roboflow")
            self._flash(f"Roboflow: {len(polygons)} polygon ditemukan ✓")
            log.info("trigger_sam_auto_current: selesai")
            bridge.deleteLater()

        def _on_error(msg: str) -> None:
            self._canvas.set_loading(False)
            self._flash("Roboflow: Gagal ❌")
            QMessageBox.warning(self, "Roboflow Error", msg)
            bridge.deleteLater()

        bridge.result.connect(_on_result)
        bridge.error.connect(_on_error)

        def _run() -> None:
            try:
                # Call Roboflow batch script with environment variables
                log.info(
                    "trigger_sam_auto_current: memanggil Roboflow untuk %s", image_path
                )
                env = os.environ.copy()
                env.update({
                    "ROBOFLOW_API_KEY": ROBOFLOW_API_KEY,
                    "ROBOFLOW_PROJECT_ID": ROBOFLOW_PROJECT_ID,
                    "ROBOFLOW_PROJECT_VERSION": ROBOFLOW_PROJECT_VERSION,
                    "ROBOFLOW_RESULTS_DIR": str(ROBOFLOW_RESULTS_DIR),
                })
                result = subprocess.run(
                    [str(ROBOFLOW_BATCH_FILE), image_path],
                    capture_output=True,
                    text=True,
                    timeout=ROBOFLOW_TIMEOUT,
                    env=env,
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    bridge.error.emit(f"Roboflow gagal: {error_msg}")
                    return

                # Update status message
                QTimer.singleShot(
                    0, lambda: self._flash("Roboflow: Menerima hasil...", ms=60000)
                )

                # Poll for result file
                start_time = time.time()
                elapsed = 0
                while not result_json_path.exists():
                    if time.time() - start_time > ROBOFLOW_TIMEOUT:
                        bridge.error.emit(
                            "Timeout menunggu hasil Roboflow (lebih dari 3 menit)"
                        )
                        return
                    time.sleep(POLL_INTERVAL)
                    elapsed += POLL_INTERVAL
                    # Update status every 2 seconds
                    if int(elapsed) % 2 == 0:
                        QTimer.singleShot(
                            0,
                            lambda e=elapsed: self._flash(
                                f"Roboflow: Menunggu hasil... ({int(e)}s)", ms=60000
                            ),
                        )

                # Read and convert result
                QTimer.singleShot(
                    0, lambda: self._flash("Roboflow: Memproses hasil...", ms=60000)
                )
                with open(result_json_path, encoding="utf-8") as f:
                    roboflow_data = json.load(f)

                existing = self._canvas.get_polygons()
                start_id = max((p.instance_id for p in existing), default=0) + 1

                polygons = roboflow_to_polygons(
                    roboflow_data, instance_id_start=start_id
                )

                # Filter by confidence
                filtered = [p for p in polygons if p.confidence >= self._confidence]
                log.info(
                    "trigger_sam_auto_current: %d polygon lolos filter, mengirim ke UI...",
                    len(filtered),
                )
                bridge.result.emit(filtered)

            except subprocess.TimeoutExpired:
                bridge.error.emit("Timeout menjalankan Roboflow (lebih dari 3 menit)")
            except Exception as exc:
                log.error("trigger_sam_auto_current: error — %s", exc, exc_info=True)
                bridge.error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def trigger_sam_auto_all(self) -> None:
        images = self._img_manager.get_all()
        if not images:
            QMessageBox.information(
                self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu."
            )
            return
        unannotated = [e for e in images if not e.annotated]
        log.info(
            "trigger_sam_auto_all: %d total gambar, %d belum dianotasi",
            len(images),
            len(unannotated),
        )

        dlg = _AutoAllDialog(len(unannotated), self._confidence, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        entries = unannotated if dlg.skip_annotated() else images
        confidence = dlg.confidence()
        log.info(
            "trigger_sam_auto_all: memproses %d gambar dengan confidence >= %.2f",
            len(entries),
            confidence,
        )

        progress_dlg = QProgressDialog(
            "Memproses gambar…", "Batal", 0, len(entries), self
        )
        progress_dlg.setWindowModality(Qt.WindowModal)
        progress_dlg.show()

        # Create temp file with image list
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            image_list_file = f.name
            for entry in entries:
                f.write(entry.filepath + "\n")

        # Clean up old results
        for entry in entries:
            result_json_path = (
                ROBOFLOW_RESULTS_DIR / f"{Path(entry.filepath).stem}.json"
            )
            if result_json_path.exists():
                result_json_path.unlink()

        bridge = _ThreadBridge(self)
        stop_event = threading.Event()
        total_polygons = 0

        def _on_result(polygons: list) -> None:
            nonlocal total_polygons
            total_polygons += len(polygons)

        def _on_error(msg: str) -> None:
            progress_dlg.close()
            QMessageBox.warning(self, "Roboflow Batch Error", msg)
            bridge.deleteLater()

        bridge.result.connect(_on_result)
        bridge.error.connect(_on_error)

        def _run() -> None:
            nonlocal total_polygons
            try:
                # Call Roboflow batch script with environment variables
                batch_file = ROBOFLOW_ROOT / "run_batch.bat"
                log.info(
                    "trigger_sam_auto_all: memanggil Roboflow batch untuk %d gambar",
                    len(entries),
                )

                env = os.environ.copy()
                env.update({
                    "ROBOFLOW_API_KEY": ROBOFLOW_API_KEY,
                    "ROBOFLOW_PROJECT_ID": ROBOFLOW_PROJECT_ID,
                    "ROBOFLOW_PROJECT_VERSION": ROBOFLOW_PROJECT_VERSION,
                    "ROBOFLOW_RESULTS_DIR": str(ROBOFLOW_RESULTS_DIR),
                })

                process = subprocess.Popen(
                    [str(batch_file), image_list_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

                # Wait for process to complete with timeout
                timeout_per_image = ROBOFLOW_TIMEOUT
                total_timeout = timeout_per_image * len(entries)
                start_time = time.time()

                # Poll for completion and results
                completed = set()
                while len(completed) < len(entries):
                    if stop_event.is_set():
                        process.terminate()
                        bridge.error.emit("Batch processing dibatalkan")
                        return

                    if time.time() - start_time > total_timeout:
                        process.terminate()
                        bridge.error.emit(
                            f"Timeout batch processing (lebih dari {total_timeout} detik)"
                        )
                        return

                    # Check if process has errored
                    if process.poll() is not None and process.returncode != 0:
                        stderr = process.stderr.read() if process.stderr else ""
                        bridge.error.emit(f"Roboflow batch gagal: {stderr}")
                        return

                    # Check for completed results
                    for idx, entry in enumerate(entries):
                        if entry.filepath in completed:
                            continue

                        result_json_path = (
                            ROBOFLOW_RESULTS_DIR / f"{Path(entry.filepath).stem}.json"
                        )
                        if result_json_path.exists():
                            try:
                                with open(result_json_path, encoding="utf-8") as f:
                                    roboflow_data = json.load(f)

                                existing = self._img_manager.load_annotation(entry)
                                start_id = (
                                    max((p.instance_id for p in existing), default=0)
                                    + 1
                                )

                                polygons = roboflow_to_polygons(
                                    roboflow_data, instance_id_start=start_id
                                )
                                filtered = [
                                    p for p in polygons if p.confidence >= confidence
                                ]

                                # Compute occlusion levels for Roboflow polygons
                                if filtered:
                                    from src.auto_labeler import compute_occlusion_levels
                                    image_shape = (entry.height, entry.width)
                                    compute_occlusion_levels(filtered, image_shape)

                                # Save annotation
                                self._img_manager.save_annotation(
                                    entry, existing + filtered
                                )
                                entry.auto_labeled = True

                                completed.add(entry.filepath)
                                log.info(
                                    "trigger_sam_auto_all: %s selesai, %d polygon",
                                    entry.filename,
                                    len(filtered),
                                )

                                # Update progress dialog (must be in main thread)
                                progress_dlg.setValue(len(completed))
                                progress_dlg.setLabelText(
                                    f"Memproses {entry.filename}…"
                                )

                            except Exception as e:
                                log.error(
                                    "Error processing result for %s: %s",
                                    entry.filename,
                                    e,
                                )

                    time.sleep(POLL_INTERVAL)

                # Wait for process to finish
                process.wait()
                bridge.result.emit([])
                bridge.deleteLater()

                # Final update in main thread
                QTimer.singleShot(
                    0,
                    lambda: self._on_batch_finished(
                        total_polygons, len(entries), progress_dlg
                    ),
                )

            except Exception as exc:
                log.error("trigger_sam_auto_all: error — %s", exc, exc_info=True)
                bridge.error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()
        progress_dlg.canceled.connect(lambda: stop_event.set())

    def accept_all_auto(self) -> None:
        self._canvas.accept_all_auto()
        self._update_instance_list(self._canvas.get_polygons())
        self._update_thumbnail()
        self._update_status_bar()

    def reject_selected_auto(self) -> None:
        pending_ids = [p.instance_id for p in self._canvas._pending_polygons]
        for pid in pending_ids:
            self._canvas.reject_auto_polygon(pid)
        self._update_instance_list(self._canvas.get_polygons())

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export_coco_splits(self) -> None:
        if not self._img_manager.get_all():
            QMessageBox.information(self, "Export", "Tidak ada gambar untuk diekspor.")
            return
        log.info(
            "export_coco_splits: mulai export ke %s", self._config.paths.output_coco
        )
        self._img_manager.generate_split()
        out_dir = str(self._config.paths.output_coco)
        self._exporter.export_coco_splits(self._img_manager, out_dir)
        images = self._img_manager.get_all()
        counts: dict[str, int] = {}
        for e in images:
            if e.split:
                counts[e.split] = counts.get(e.split, 0) + 1
        summary = "\n".join(
            f"  {s}: {counts.get(s, 0)} gambar" for s in ("train", "val", "test")
        )
        log.info("export_coco_splits selesai: %s", summary.replace("\n", ", "))
        QMessageBox.information(
            self,
            "Export COCO Selesai",
            f"File disimpan di:\n{out_dir}\n\nRingkasan split:\n{summary}",
        )

    def export_voc(self) -> None:
        if not self._img_manager.get_all():
            QMessageBox.information(self, "Export", "Tidak ada gambar untuk diekspor.")
            return
        log.info(
            "export_voc: mulai export VOC XML ke %s", self._config.paths.output_voc
        )
        out_dir = str(self._config.paths.output_voc)
        written = self._exporter.export_voc_xml(self._img_manager, out_dir)
        log.info("export_voc selesai: %d file XML ditulis", len(written))
        QMessageBox.information(
            self,
            "Export XML Selesai",
            f"{len(written)} file XML disimpan di:\n{out_dir}",
        )

    def export_visualized(self) -> None:
        if not self._img_manager.get_all():
            QMessageBox.information(self, "Export", "Tidak ada gambar untuk diekspor.")
            return
        log.info(
            "export_visualized: mulai export gambar visualisasi ke %s",
            self._config.paths.output_visualized,
        )
        out_dir = str(self._config.paths.output_visualized)
        saved = self._visualizer.export_all(self._img_manager, out_dir)
        log.info("export_visualized selesai: %d gambar disimpan", len(saved))
        QMessageBox.information(
            self,
            "Export Visualized Selesai",
            f"{len(saved)} gambar disimpan di:\n{out_dir}",
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _load_image_to_canvas(self, entry: ImageEntry) -> None:
        self._current_entry = entry
        polygons = self._img_manager.load_annotation(entry)
        log.debug(
            "load_image: %s — %d polygon dimuat dari anotasi",
            entry.filename,
            len(polygons),
        )
        self._canvas.load_image(entry, polygons)
        self._update_instance_list(polygons)
        self._update_thumbnail()
        self._update_stats()
        self._update_status_bar()

    def _on_annotation_changed(self, polygons: list[Polygon]) -> None:
        if self._current_entry:
            self._current_entry.annotated = bool(polygons)
        self._update_instance_list(polygons)
        self._update_thumbnail()
        self._update_status_bar()

    def _on_has_unsaved_changes(self, has_changes: bool) -> None:
        """Handle unsaved changes signal from canvas."""
        self._has_unsaved_changes = has_changes
        self._update_sam_controls()

    def _on_point_clicked(self, x: float, y: float) -> None:
        if not self._sam_ready() or self._current_entry is None:
            return
        image = self._read_image_rgb(self._current_entry.filepath)
        if image is None:
            return
        self._canvas.set_loading(True)
        existing = self._canvas.get_polygons()
        start_id = max((p.instance_id for p in existing), default=0) + 1
        confidence = self._confidence
        labeler = self._auto_labeler

        bridge = _ThreadBridge(self)
        bridge.result.connect(
            lambda polys: (
                self._canvas.set_loading(False),
                self._on_sam_result(polys, (x, y)),  # Pass the original click point
                bridge.deleteLater(),
            )
        )
        bridge.error.connect(
            lambda msg: (
                self._canvas.set_loading(False),
                QMessageBox.warning(self, "SAM Error", msg),
                bridge.deleteLater(),
            )
        )

        def _run() -> None:
            try:
                polygons = labeler.predict_from_point(
                    image, (x, y), instance_id_start=start_id
                )
                bridge.result.emit([p for p in polygons if p.confidence >= confidence])
            except Exception as exc:
                log.error("predict_from_point: error — %s", exc, exc_info=True)
                bridge.error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_point_refine_clicked(self, instance_id: int, x: float, y: float, prompt_points: list) -> None:
        """Handle refinement click on existing polygon."""
        if not self._sam_ready() or self._current_entry is None:
            return
        image = self._read_image_rgb(self._current_entry.filepath)
        if image is None:
            return
        self._canvas.set_loading(True)
        confidence = self._confidence
        labeler = self._auto_labeler

        bridge = _ThreadBridge(self)
        bridge.result.connect(
            lambda polys: (
                self._canvas.set_loading(False),
                self._on_sam_refine_result(instance_id, polys),
                bridge.deleteLater(),
            )
        )
        bridge.error.connect(
            lambda msg: (
                self._canvas.set_loading(False),
                QMessageBox.warning(self, "SAM Refine Error", msg),
                bridge.deleteLater(),
            )
        )

        def _run() -> None:
            try:
                # Use multi-point prompt for refinement
                # All prompt points are positive (part of the object)
                # The latest click is already the last element in prompt_points
                polygons = labeler.predict_from_point(
                    image, (x, y),
                    negative_points=None,
                    positive_points=prompt_points,  # Send all prompt points including current click
                    instance_id_start=instance_id
                )
                bridge.result.emit([p for p in polygons if p.confidence >= confidence])
            except Exception as exc:
                log.error("predict_from_point refine: error — %s", exc, exc_info=True)
                bridge.error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_sam_refine_result(self, instance_id: int, polygons: list[Polygon]) -> None:
        """Handle SAM refinement result by updating the existing polygon."""
        if not polygons:
            return
        # Find and update the existing polygon
        for p in self._canvas._polygons:
            if p.instance_id == instance_id:
                new_poly = polygons[0]
                # Update the polygon with refined points
                p.points = new_poly.points
                p.confidence = new_poly.confidence
                # Keep the instance_id, source, etc.
                self._canvas._trigger_occlusion_recompute()
                break
        self._update_instance_list(self._canvas.get_polygons())
        self._update_thumbnail()
        # Enable Done button for finalizing
        self._update_sam_controls()

    def finish_refinement(self) -> None:
        """Finish the current refinement session."""
        # Accept pending polygons first (move from _pending_polygons to _polygons)
        self.accept_all_auto()
        # Then clear refinement state
        self._canvas.finish_refinement()
        self._update_sam_controls()
        self._flash("Polygon selesai ✓")

    def _on_box_selected(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self._sam_ready() or self._current_entry is None:
            return
        image = self._read_image_rgb(self._current_entry.filepath)
        if image is None:
            return
        self._canvas.set_loading(True)
        existing = self._canvas.get_polygons()
        start_id = max((p.instance_id for p in existing), default=0) + 1
        confidence = self._confidence
        labeler = self._auto_labeler

        bridge = _ThreadBridge(self)
        bridge.result.connect(
            lambda polys: (
                self._canvas.set_loading(False),
                self._on_sam_result(polys),
                bridge.deleteLater(),
            )
        )
        bridge.error.connect(
            lambda msg: (
                self._canvas.set_loading(False),
                QMessageBox.warning(self, "SAM Error", msg),
                bridge.deleteLater(),
            )
        )

        def _run() -> None:
            try:
                polygons = labeler.predict_from_box(
                    image, (x1, y1, x2, y2), instance_id_start=start_id
                )
                bridge.result.emit([p for p in polygons if p.confidence >= confidence])
            except Exception as exc:
                log.error("predict_from_box: error — %s", exc, exc_info=True)
                bridge.error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_sam_result(self, polygons: list[Polygon], click_point: tuple[float, float] | None = None, source: str = "sam_point") -> None:
        if polygons:
            self._canvas.inject_polygons(polygons)
            all_polys = self._canvas._polygons + self._canvas._pending_polygons
            self._update_instance_list(all_polys)

            # For Roboflow polygons, compute occlusion levels and auto-accept
            if source == "roboflow":
                if self._current_entry:
                    image_shape = (self._current_entry.height, self._current_entry.width)
                    # Compute occlusion for Roboflow polygons
                    from src.auto_labeler import compute_occlusion_levels
                    # Only compute for polygons that don't have occlusion_level yet
                    polygons_to_compute = [p for p in polygons if p.occlusion_level is None]
                    if polygons_to_compute:
                        compute_occlusion_levels(polygons_to_compute, image_shape)
                        # Auto-accept Roboflow polygons since occlusion is computed
                        self._canvas.accept_all_auto()
                        self._update_instance_list(self._canvas.get_polygons())
                        self._update_thumbnail()
                self._flash(f"Roboflow: {len(polygons)} polygon ditemukan ✓")
            else:
                # For SAM point prediction, auto-start refinement mode
                new_poly = polygons[0]
                # Use the original click point for refinement start
                refine_point = click_point if click_point else (new_poly.points[0] if new_poly.points else (0, 0))
                self._canvas.start_refinement(new_poly.instance_id, refine_point)
                self._update_sam_controls()
                self._flash(f"Polygon #{new_poly.instance_id} dibuat. Klik di dalam untuk refine, tekan Done untuk final")

    def _on_batch_result(self, filename: str, polygons: list[Polygon]) -> None:
        entry_map = {e.filename: e for e in self._img_manager.get_all()}
        if filename not in entry_map:
            return
        entry = entry_map[filename]
        existing = self._img_manager.load_annotation(entry)
        max_id = max((p.instance_id for p in existing), default=0)
        for i, p in enumerate(polygons):
            p.instance_id = max_id + i + 1
        self._img_manager.save_annotation(entry, existing + polygons)
        entry.auto_labeled = True

    def _on_batch_finished(
        self, total_poly: int, total_images: int, dlg: QProgressDialog
    ) -> None:
        log.info("batch selesai: %d polygon dari %d gambar", total_poly, total_images)
        dlg.close()
        QMessageBox.information(
            self,
            "Auto-label Selesai",
            f"Auto-label selesai: {total_images} gambar diproses, "
            f"{total_poly} polygon digenerate.\n"
            "Polygon menunggu review di setiap gambar sebelum export.",
        )
        if self._current_entry:
            self._load_image_to_canvas(self._current_entry)
        self._update_stats()

    def _update_status_bar(self) -> None:
        images = self._img_manager.get_all()
        if not images or self._current_entry is None:
            self._sb_file.setText("Belum ada gambar")
            self._sb_mode.setText("")
            self._sb_zoom.setText("Zoom: 100%")
            return
        entry = self._current_entry
        total = len(images)
        pos = entry.index + 1
        n_inst = len(self._canvas.get_polygons())
        mode = self._canvas._mode.value.upper()
        zoom_pct = int(self._canvas._zoom_scale * 100)

        # Show refinement status if active
        if self._canvas.is_refinement_active():
            refine_id = self._canvas.get_active_refinement_id()
            n_prompts = len(self._canvas._refinement_prompt_points)
            self._sb_mode.setText(f"{mode} | Refining #{refine_id} ({n_prompts} pts)")
        else:
            self._sb_mode.setText(mode)

        self._sb_file.setText(f"◀ {entry.filename} ({pos}/{total}) ▶ | {n_inst} inst.")
        self._sb_zoom.setText(f"Zoom: {zoom_pct}%")

    def _update_instance_list(self, polygons: list[Polygon]) -> None:
        self._inst_list.clear()
        pending_ids = {p.instance_id for p in self._canvas._pending_polygons}
        all_display = self._canvas._polygons + self._canvas._pending_polygons
        MAX_ROWS = 50

        for p in all_display[:MAX_ROWS]:
            is_pending = p.instance_id in pending_ids
            is_auto = p.source in {"sam_point", "sam_auto", "roboflow"}
            src = "manual" if p.source == "manual" else p.source.replace("_", " ")
            badge = ""
            if is_auto:
                # Use proper label based on source
                if p.source == "roboflow":
                    badge = f" Roboflow {p.confidence:.2f}"
                else:
                    badge = f" SAM {p.confidence:.2f}"
                if is_pending:
                    badge += " ⚠️"

            item = QListWidgetItem()
            c = (
                p.color
                if isinstance(p.color, list) and len(p.color) == 3
                else [128, 128, 128]
            )
            item.setBackground(QColor(c[0], c[1], c[2], 80))
            item.setData(Qt.UserRole, p.instance_id)
            self._inst_list.addItem(item)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 0, 2, 0)
            lbl = QLabel(f"#{p.instance_id} {src}{badge}")
            del_btn = QPushButton("×")
            del_btn.setFixedWidth(20)
            del_btn.setFlat(True)
            del_btn.setStyleSheet("color: red;")
            instance_id = p.instance_id
            del_btn.clicked.connect(
                lambda _checked, iid=instance_id: self._delete_instance(iid)
            )
            row_layout.addWidget(lbl, 1)
            row_layout.addWidget(del_btn)
            item.setSizeHint(row_widget.sizeHint())
            self._inst_list.setItemWidget(item, row_widget)

        overflow = len(all_display) - MAX_ROWS
        if overflow > 0:
            self._overflow_lbl.setText(f"… and {overflow} more")
            self._overflow_lbl.show()
        else:
            self._overflow_lbl.hide()

        self._inst_list.scrollToBottom()

    def _delete_instance(self, instance_id: int) -> None:
        pending_ids = {p.instance_id for p in self._canvas._pending_polygons}
        if instance_id in pending_ids:
            self._canvas.reject_auto_polygon(instance_id)
        else:
            prev_selected = self._canvas._selected_id
            self._canvas._selected_id = instance_id
            self._canvas.delete_selected()
            self._canvas._selected_id = prev_selected
        self._update_instance_list(self._canvas.get_polygons())
        self._update_thumbnail()

    def _update_thumbnail(self) -> None:
        if self._current_entry is None:
            self._thumb.clear()
            return
        try:
            polygons = self._canvas.get_polygons()
            pixmap = self._visualizer.render_thumbnail(
                self._current_entry.filepath, polygons
            )
            self._thumb.setPixmap(pixmap)
        except Exception:
            self._thumb.clear()

    def _update_stats(self) -> None:
        images = self._img_manager.get_all()
        if not images:
            self._lbl_dataset.setText("📁 Belum ada gambar")
            self._progress_bar.setMaximum(1)
            self._progress_bar.setValue(0)
            self._lbl_auto.setText("🤖 0 auto-labeled")
            return
        prog = self._img_manager.get_progress()
        self._lbl_dataset.setText(f"📁 Dataset: {prog.total} gambar")
        self._progress_bar.setMaximum(prog.total)
        self._progress_bar.setValue(prog.annotated)
        self._lbl_auto.setText(f"🤖 {prog.auto_labeled} auto-labeled")

    def _update_sam_controls(self) -> None:
        # Roboflow (Auto) always ready
        can_interact = self._model_load_attempted

        # Point/Box buttons require MobileSAM
        sam_ready = self._sam_ready()
        if sam_ready:
            self._btn_point.show()
            self._btn_box.show()
            self._btn_point.setEnabled(can_interact)
            self._btn_box.setEnabled(can_interact)
        else:
            self._btn_point.hide()
            self._btn_box.hide()

        # Enable Auto, Accept, Reject buttons (Roboflow always available)
        for btn in (self._btn_auto_cur, self._btn_accept, self._btn_reject):
            btn.setEnabled(can_interact)
        self._conf_slider.setEnabled(can_interact)
        self._act_auto_label.setEnabled(can_interact)
        self._act_auto_all.setEnabled(can_interact)

        # Update manual control buttons based on state
        is_draw_mode = (self._canvas._mode == CanvasMode.DRAW)
        is_point_mode = (self._canvas._mode == CanvasMode.SAM_POINT)
        is_refining = self._canvas.is_refinement_active()
        is_manual_active = is_draw_mode or is_point_mode

        # Draw/Select buttons: visible when NOT in manual mode
        self._btn_manual_draw.setVisible(not is_manual_active)
        self._btn_manual_select.setVisible(not is_manual_active)

        # Save/Cancel buttons: visible when in manual mode
        self._btn_manual_save.setVisible(is_manual_active)
        self._btn_manual_cancel.setVisible(is_manual_active)

        # Save button: enabled only when there are changes (points clicked or polygon created)
        has_changes = self._has_unsaved_changes or is_refining
        self._btn_manual_save.setEnabled(has_changes)
        self._btn_manual_cancel.setEnabled(is_manual_active)

        # Update Undo/Redo buttons based on canvas undo stack
        can_undo = len(self._canvas._undo_stack) > 0
        can_redo = False  # TODO: implement redo stack
        self._btn_undo.setEnabled(can_undo)
        self._btn_redo.setEnabled(can_redo)

    def trigger_manual_draw(self) -> None:
        """Switch to Draw mode for manual polygon annotation."""
        # Save initial state snapshot
        self._initial_polygons_snapshot = copy.deepcopy(self._canvas._polygons)
        self._has_unsaved_changes = False

        self._set_mode(CanvasMode.DRAW)
        self._update_sam_controls()
        self._flash("Mode Draw: Klik untuk menambahkan titik, double-klik untuk menutup polygon")

    def trigger_manual_select(self) -> None:
        """Switch to Select (SAM Point) mode for manual point selection."""
        if self._canvas.is_refinement_active():
            # If already in select mode, this is handled by Save button
            return
        # Save initial state snapshot
        self._initial_polygons_snapshot = copy.deepcopy(self._canvas._polygons)
        self._has_unsaved_changes = False

        self.trigger_sam_point()
        self._update_sam_controls()

    def finish_manual_save(self) -> None:
        """Save the current manual select/draw operation."""
        if self._canvas.is_refinement_active():
            # For Point mode with active refinement
            self.finish_refinement()
        elif self._canvas._mode == CanvasMode.DRAW:
            # For Draw mode - accept current polygon and exit
            self._canvas.set_mode(CanvasMode.VIEW)
            self._update_sam_controls()
            self._flash("Perubahan disimpan ✓")
        elif self._canvas._mode == CanvasMode.SAM_POINT:
            # For Point mode without refinement - just exit
            self._canvas.set_mode(CanvasMode.VIEW)
            self._update_sam_controls()
            self._flash("Perubahan disimpan ✓")

        # Reset unsaved changes flag
        self._has_unsaved_changes = False
        self._initial_polygons_snapshot = []

    def cancel_manual_select(self) -> None:
        """Cancel the current manual select/draw operation."""
        # Check if there are unsaved changes
        if self._has_unsaved_changes or self._canvas.is_refinement_active():
            reply = QMessageBox.question(
                self, "Peringatan",
                "Ada perubahan yang belum disimpan. Yakin ingin membatalkan?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        if self._canvas.is_refinement_active():
            # Cancel refinement - discard the current polygon
            self._canvas.finish_refinement()
            # Remove the last created polygon (the one being refined)
            if self._canvas._polygons:
                for i, p in enumerate(self._canvas._polygons):
                    if p.instance_id == self._canvas.get_active_refinement_id():
                        self._canvas._polygons.pop(i)
                        break
            self._update_sam_controls()
            self._flash("Polygon dibatalkan")
        elif self._canvas._mode == CanvasMode.DRAW:
            # Cancel Draw mode - discard active points
            self._canvas._active_points = []
            self._canvas.set_mode(CanvasMode.VIEW)
            self._update_sam_controls()
            self._flash("Mode Draw dibatalkan")
        elif self._canvas._mode == CanvasMode.SAM_POINT:
            # Cancel Point mode
            self._canvas.set_mode(CanvasMode.VIEW)
            self._update_sam_controls()
            self._flash("Mode Select dibatalkan")

        # Reset unsaved changes flag
        self._has_unsaved_changes = False
        self._initial_polygons_snapshot = []

    def undo_action(self) -> None:
        """Perform undo action."""
        self._canvas.undo()
        self._update_sam_controls()
        self._update_instance_list(self._canvas.get_polygons())
        self._update_thumbnail()
        self._flash("Undo")

    def redo_action(self) -> None:
        """Perform redo action."""
        # TODO: implement redo
        self._flash("Redo belum diimplementasi")

    def _set_mode(self, mode: CanvasMode) -> None:
        self._canvas.set_mode(mode)
        self._update_sam_controls()
        self._update_status_bar()

    def _sam_ready(self) -> bool:
        # Check if MobileSAM is available for Point/Box features
        if not _HAS_SAM or self._auto_labeler is None:
            return False
        return self._auto_labeler.is_loaded()

    def _notify_sam_not_ready(self) -> None:
        if not _HAS_SAM:
            QMessageBox.warning(
                self,
                "MobileSAM Tidak Tersedia",
                "Fitur Point/Box prompting memerlukan MobileSAM.\n\n"
                "Pastikan dependensi berikut terinstall:\n"
                "  pip install mobile-sam torch torchvision",
            )
            return
        checkpoint = Path(str(self._config.paths.sam_checkpoint))
        if not checkpoint.exists():
            QMessageBox.warning(
                self,
                "Checkpoint MobileSAM Tidak Ditemukan",
                f"File MobileSAM tidak ditemukan:\n{checkpoint}\n\n"
                "Download (~40 MB) dari:\n"
                "  https://github.com/ChaoningZhang/MobileSAM\n"
                "  (file: weights/mobile_sam.pt)\n\n"
                f"Letakkan file di:\n  {checkpoint.parent}\\mobile_sam.pt",
            )
        else:
            QMessageBox.information(
                self,
                "SAM Sedang Dimuat",
                "Model SAM masih dalam proses loading.\n"
                "Tunggu beberapa saat lalu coba lagi.",
            )

    def _read_image_rgb(self, filepath: str) -> "np.ndarray | None":
        image = cv2.imread(filepath)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_model_async(self) -> None:
        bridge = _ThreadBridge(self)

        def _on_ready(_polys: list) -> None:
            log.info("_load_model_async: update SAM controls di main thread")
            self._model_load_attempted = True
            self._update_sam_controls()
            bridge.deleteLater()

        bridge.result.connect(_on_ready)

        def _load() -> None:
            if self._auto_labeler:
                self._auto_labeler.load_model()
                log.info(
                    "_load_model_async: model siap, is_loaded=%s",
                    self._auto_labeler.is_loaded(),
                )
            bridge.result.emit([])

        threading.Thread(target=_load, daemon=True).start()

    def _on_conf_changed(self, value: int) -> None:
        self._confidence = value / 100.0
        self._conf_lbl.setText(f"{self._confidence:.2f}")

    def _on_instance_clicked(self, item: QListWidgetItem) -> None:
        instance_id = item.data(Qt.UserRole)
        if instance_id is not None:
            self._canvas._selected_id = instance_id
            if self._canvas._mode != CanvasMode.SELECT:
                self._canvas._mode = CanvasMode.SELECT
            self._canvas.update()
            self._update_status_bar()

    def _handle_escape(self) -> None:
        """Handle Escape key - cancel manual mode or reject pending."""
        is_draw_mode = (self._canvas._mode == CanvasMode.DRAW)
        is_point_mode = (self._canvas._mode == CanvasMode.SAM_POINT)

        if is_draw_mode or is_point_mode or self._canvas.is_refinement_active():
            self.cancel_manual_select()
        else:
            self.reject_selected_auto()

    def _handle_return_key(self) -> None:
        """Handle Return key - save manual mode or refinement if active, otherwise accept all."""
        is_draw_mode = (self._canvas._mode == CanvasMode.DRAW)
        is_point_mode = (self._canvas._mode == CanvasMode.SAM_POINT)

        if is_draw_mode or is_point_mode:
            self.finish_manual_save()
        elif self._canvas.is_refinement_active():
            self.finish_refinement()
        else:
            self.accept_all_auto()

    def _flash(self, msg: str, ms: int = 2000) -> None:
        old = self._sb_file.text()
        self._sb_file.setText(msg)
        QTimer.singleShot(ms, lambda: self._sb_file.setText(old))


# ---------------------------------------------------------------------------
# Auto-Label All dialog
# ---------------------------------------------------------------------------


class _AutoAllDialog(QDialog):
    def __init__(
        self, unannotated_count: int, default_confidence: float, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Label Semua Gambar")
        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(f"Akan memproses {unannotated_count} gambar yang belum dianotasi.")
        )

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence threshold:"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(50, 99)
        self._slider.setValue(int(default_confidence * 100))
        self._conf_lbl = QLabel(f"{default_confidence:.2f}")
        self._slider.valueChanged.connect(
            lambda v: self._conf_lbl.setText(f"{v / 100:.2f}")
        )
        conf_row.addWidget(self._slider)
        conf_row.addWidget(self._conf_lbl)
        layout.addLayout(conf_row)

        self._skip_cb = QCheckBox("Skip gambar yang sudah dianotasi")
        self._skip_cb.setChecked(True)
        layout.addWidget(self._skip_cb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Mulai")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def confidence(self) -> float:
        return self._slider.value() / 100.0

    def skip_annotated(self) -> bool:
        return self._skip_cb.isChecked()
