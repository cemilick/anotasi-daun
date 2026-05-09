from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

        if _HAS_SAM:
            self._auto_labeler = AutoLabeler(config)
            log.info("MainWindow: memulai load model AutoLabeler...")
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
        tb1 = self.addToolBar("Main")
        tb1.setObjectName("tb_main")

        self._act_open = QAction("Open Folder", self)
        self._act_save = QAction("Save", self)
        tb1.addAction(self._act_open)
        tb1.addAction(self._act_save)
        tb1.addSeparator()

        self._act_auto_label = QAction("✨ Auto Label", self)
        self._act_auto_all = QAction("⚡ Auto All", self)
        tb1.addAction(self._act_auto_label)
        tb1.addAction(self._act_auto_all)
        tb1.addSeparator()

        self._act_draw = QAction("Draw", self)
        self._act_select = QAction("Select", self)
        self._act_draw.setCheckable(True)
        self._act_select.setCheckable(True)
        tb1.addAction(self._act_draw)
        tb1.addAction(self._act_select)

        tb2 = self.addToolBar("Export")
        tb2.setObjectName("tb_export")

        self._act_exp_coco = QAction("Export COCO Splits", self)
        self._act_exp_voc = QAction("Export XML", self)
        self._act_exp_vis = QAction("Export Visualized", self)
        tb2.addAction(self._act_exp_coco)
        tb2.addAction(self._act_exp_voc)
        tb2.addAction(self._act_exp_vis)
        tb2.addSeparator()

        self._act_zoom_in = QAction("+", self)
        self._act_zoom_out = QAction("-", self)
        self._act_zoom_fit = QAction("Fit", self)
        tb2.addAction(self._act_zoom_in)
        tb2.addAction(self._act_zoom_out)
        tb2.addAction(self._act_zoom_fit)

        self._act_open.triggered.connect(self.open_folder)
        self._act_save.triggered.connect(self.save_current)
        self._act_auto_label.triggered.connect(self.trigger_sam_auto_current)
        self._act_auto_all.triggered.connect(self.trigger_sam_auto_all)
        self._act_draw.triggered.connect(lambda: self._set_mode(CanvasMode.DRAW))
        self._act_select.triggered.connect(lambda: self._set_mode(CanvasMode.SELECT))
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

        # SAM Controls
        self._sam_group = QGroupBox("SAM Controls")
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
        sc("A", self.trigger_sam_point)
        sc("B", self.trigger_sam_box)
        sc("D", lambda: self._set_mode(CanvasMode.DRAW))
        sc("V", lambda: self._set_mode(CanvasMode.SELECT))
        sc("Return", self.accept_all_auto)
        sc("Escape", self._handle_escape)
        sc("Delete", self._canvas.delete_selected)
        sc("Ctrl+Z", self._canvas.undo)
        sc("+", self._canvas.zoom_in)
        sc("-", self._canvas.zoom_out)
        sc("F", self._canvas.reset_zoom)

    def _connect_signals(self) -> None:
        self._canvas.annotation_changed.connect(self._on_annotation_changed)
        self._canvas.point_clicked.connect(self._on_point_clicked)
        self._canvas.box_selected.connect(self._on_box_selected)

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
        log.info("save_current: %d polygon disimpan untuk %s", len(polygons), self._current_entry.filename)
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
        log.info("next_image: navigasi ke %s (%d/%d)", entry.filename, entry.index + 1, len(images))
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
        log.info("prev_image: navigasi ke %s (%d/%d)", entry.filename, entry.index + 1, len(images))
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
            QMessageBox.information(self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu.")
            return
        log.info("trigger_sam_point: mode SAM_POINT aktif, klik pada gambar untuk prediksi")
        self._set_mode(CanvasMode.SAM_POINT)

    def trigger_sam_box(self) -> None:
        if not self._sam_ready():
            self._notify_sam_not_ready()
            return
        if self._current_entry is None:
            QMessageBox.information(self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu.")
            return
        log.info("trigger_sam_box: mode SAM_BOX aktif, seret kotak pada gambar untuk prediksi")
        self._set_mode(CanvasMode.SAM_BOX)

    def trigger_sam_auto_current(self) -> None:
        if not self._sam_ready():
            self._notify_sam_not_ready()
            return
        if self._current_entry is None:
            QMessageBox.information(self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu.")
            return
        log.info("trigger_sam_auto_current: mulai auto-label untuk %s", self._current_entry.filename)
        self._canvas.set_loading(True)
        image = self._read_image_rgb(self._current_entry.filepath)
        if image is None:
            self._canvas.set_loading(False)
            return
        existing = self._canvas.get_polygons()
        start_id = max((p.instance_id for p in existing), default=0) + 1
        confidence = self._confidence
        labeler = self._auto_labeler

        def _run() -> None:
            try:
                polygons = labeler.predict_automatic(image, instance_id_start=start_id)
                filtered = [p for p in polygons if p.confidence >= confidence]
                log.info("trigger_sam_auto_current: %d polygon lolos filter confidence %.2f", len(filtered), confidence)
                log.info("trigger_sam_auto_current: mengirim hasil ke UI thread...")
                def _done() -> None:
                    log.info("trigger_sam_auto_current: UI thread menerima hasil, inject polygon...")
                    self._canvas.set_loading(False)
                    self._on_sam_result(filtered)
                    log.info("trigger_sam_auto_current: selesai")
                QTimer.singleShot(0, _done)
            except Exception as exc:
                log.error("trigger_sam_auto_current: error — %s", exc, exc_info=True)
                msg = str(exc)
                def _err() -> None:
                    self._canvas.set_loading(False)
                    QMessageBox.warning(self, "SAM Error", msg)
                QTimer.singleShot(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    def trigger_sam_auto_all(self) -> None:
        if not self._sam_ready():
            self._notify_sam_not_ready()
            return
        images = self._img_manager.get_all()
        if not images:
            QMessageBox.information(self, "Belum Ada Gambar", "Buka folder gambar terlebih dahulu.")
            return
        unannotated = [e for e in images if not e.annotated]
        log.info("trigger_sam_auto_all: %d total gambar, %d belum dianotasi", len(images), len(unannotated))

        dlg = _AutoAllDialog(len(unannotated), self._confidence, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        entries = unannotated if dlg.skip_annotated() else images
        confidence = dlg.confidence()
        log.info("trigger_sam_auto_all: memproses %d gambar dengan confidence >= %.2f", len(entries), confidence)

        progress_dlg = QProgressDialog("Memproses gambar…", "Batal", 0, len(entries), self)
        progress_dlg.setWindowModality(Qt.WindowModal)
        progress_dlg.show()

        self._auto_worker = AutoLabelWorker(self._auto_labeler, entries, confidence)

        def _on_progress(cur: int, tot: int, fname: str) -> None:
            progress_dlg.setValue(cur)
            progress_dlg.setLabelText(f"Memproses {fname}…")

        self._auto_worker.progress.connect(_on_progress)
        self._auto_worker.result_ready.connect(self._on_batch_result)
        self._auto_worker.finished.connect(
            lambda n_poly: self._on_batch_finished(n_poly, len(entries), progress_dlg)
        )
        self._auto_worker.error.connect(
            lambda msg: QMessageBox.warning(self, "SAM Error", msg)
        )
        progress_dlg.canceled.connect(self._auto_worker.stop)
        self._auto_worker.start()

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
        log.info("export_coco_splits: mulai export ke %s", self._config.paths.output_coco)
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
            self, "Export COCO Selesai",
            f"File disimpan di:\n{out_dir}\n\nRingkasan split:\n{summary}",
        )

    def export_voc(self) -> None:
        if not self._img_manager.get_all():
            QMessageBox.information(self, "Export", "Tidak ada gambar untuk diekspor.")
            return
        log.info("export_voc: mulai export VOC XML ke %s", self._config.paths.output_voc)
        out_dir = str(self._config.paths.output_voc)
        written = self._exporter.export_voc_xml(self._img_manager, out_dir)
        log.info("export_voc selesai: %d file XML ditulis", len(written))
        QMessageBox.information(
            self, "Export XML Selesai",
            f"{len(written)} file XML disimpan di:\n{out_dir}",
        )

    def export_visualized(self) -> None:
        if not self._img_manager.get_all():
            QMessageBox.information(self, "Export", "Tidak ada gambar untuk diekspor.")
            return
        log.info("export_visualized: mulai export gambar visualisasi ke %s", self._config.paths.output_visualized)
        out_dir = str(self._config.paths.output_visualized)
        saved = self._visualizer.export_all(self._img_manager, out_dir)
        log.info("export_visualized selesai: %d gambar disimpan", len(saved))
        QMessageBox.information(
            self, "Export Visualized Selesai",
            f"{len(saved)} gambar disimpan di:\n{out_dir}",
        )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _load_image_to_canvas(self, entry: ImageEntry) -> None:
        self._current_entry = entry
        polygons = self._img_manager.load_annotation(entry)
        log.debug("load_image: %s — %d polygon dimuat dari anotasi", entry.filename, len(polygons))
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

        def _run() -> None:
            try:
                polygons = labeler.predict_from_point(image, (x, y), instance_id_start=start_id)
                filtered = [p for p in polygons if p.confidence >= confidence]
                def _done() -> None:
                    self._canvas.set_loading(False)
                    self._on_sam_result(filtered)
                QTimer.singleShot(0, _done)
            except Exception as exc:
                msg = str(exc)
                def _err() -> None:
                    self._canvas.set_loading(False)
                    QMessageBox.warning(self, "SAM Error", msg)
                QTimer.singleShot(0, _err)

        threading.Thread(target=_run, daemon=True).start()

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

        def _run() -> None:
            try:
                polygons = labeler.predict_from_box(
                    image, (x1, y1, x2, y2), instance_id_start=start_id
                )
                filtered = [p for p in polygons if p.confidence >= confidence]
                def _done() -> None:
                    self._canvas.set_loading(False)
                    self._on_sam_result(filtered)
                QTimer.singleShot(0, _done)
            except Exception as exc:
                msg = str(exc)
                def _err() -> None:
                    self._canvas.set_loading(False)
                    QMessageBox.warning(self, "SAM Error", msg)
                QTimer.singleShot(0, _err)

        threading.Thread(target=_run, daemon=True).start()

    def _on_sam_result(self, polygons: list[Polygon]) -> None:
        if polygons:
            self._canvas.inject_polygons(polygons)
            all_polys = self._canvas._polygons + self._canvas._pending_polygons
            self._update_instance_list(all_polys)

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
        self._sb_file.setText(
            f"◀ {entry.filename} ({pos}/{total}) ▶ | {n_inst} inst."
        )
        self._sb_mode.setText(mode)
        self._sb_zoom.setText(f"Zoom: {zoom_pct}%")

    def _update_instance_list(self, polygons: list[Polygon]) -> None:
        self._inst_list.clear()
        pending_ids = {p.instance_id for p in self._canvas._pending_polygons}
        all_display = self._canvas._polygons + self._canvas._pending_polygons
        MAX_ROWS = 50

        for p in all_display[:MAX_ROWS]:
            is_pending = p.instance_id in pending_ids
            is_sam = p.source in {"sam_point", "sam_auto"}
            src = "manual" if p.source == "manual" else p.source.replace("_", " ")
            badge = ""
            if is_sam:
                badge = f" SAM {p.confidence:.2f}"
                if is_pending:
                    badge += " ⚠️"

            item = QListWidgetItem()
            c = p.color if isinstance(p.color, list) and len(p.color) == 3 else [128, 128, 128]
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
            del_btn.clicked.connect(lambda _checked, iid=instance_id: self._delete_instance(iid))
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
        ready = self._sam_ready()
        for btn in (
            self._btn_point, self._btn_box, self._btn_auto_cur,
            self._btn_accept, self._btn_reject,
        ):
            btn.setEnabled(ready)
        self._conf_slider.setEnabled(ready)
        # Toolbar actions must also reflect readiness
        self._act_auto_label.setEnabled(ready)
        self._act_auto_all.setEnabled(ready)
        if not ready and self._auto_labeler is not None:
            self._sb_mode.setText("SAM loading…")

    def _set_mode(self, mode: CanvasMode) -> None:
        self._canvas.set_mode(mode)
        self._act_draw.setChecked(mode == CanvasMode.DRAW)
        self._act_select.setChecked(mode == CanvasMode.SELECT)
        self._update_status_bar()

    def _sam_ready(self) -> bool:
        if not _HAS_SAM or self._auto_labeler is None:
            return False
        return self._auto_labeler.is_loaded()

    def _notify_sam_not_ready(self) -> None:
        if not _HAS_SAM:
            QMessageBox.warning(
                self, "SAM Tidak Tersedia",
                "Modul auto_labeler tidak dapat dimuat.\n\n"
                "Pastikan dependensi berikut terinstall:\n"
                "  pip install segment-anything torch torchvision",
            )
            return
        checkpoint = Path(str(self._config.paths.sam_checkpoint))
        if not checkpoint.exists():
            QMessageBox.warning(
                self, "Checkpoint SAM Tidak Ditemukan",
                f"File SAM checkpoint tidak ditemukan:\n{checkpoint}\n\n"
                "Download dari:\n"
                "  https://github.com/facebookresearch/segment-anything#model-checkpoints\n\n"
                f"Letakkan file di:\n  {checkpoint.parent}/",
            )
        else:
            QMessageBox.information(
                self, "SAM Sedang Dimuat",
                "Model SAM masih dalam proses loading.\n"
                "Tunggu beberapa saat lalu coba lagi.",
            )

    def _read_image_rgb(self, filepath: str) -> "np.ndarray | None":
        image = cv2.imread(filepath)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_model_async(self) -> None:
        def _load() -> None:
            if self._auto_labeler:
                self._auto_labeler.load_model()
                log.info("_load_model_async: model siap, is_loaded=%s", self._auto_labeler.is_loaded())
                QTimer.singleShot(0, self._update_sam_controls)
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
                self._act_select.setChecked(True)
                self._act_draw.setChecked(False)
            self._canvas.update()
            self._update_status_bar()

    def _handle_escape(self) -> None:
        if self._canvas._mode == CanvasMode.DRAW and self._canvas._active_points:
            self._canvas._active_points = []
            self._canvas.update()
        else:
            self.reject_selected_auto()

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
