# Spec 06 — Main UI & Integrasi (Aplikasi Final)

## Goal

Sambungkan semua modul menjadi aplikasi PyQt5 yang fungsional dengan toolbar, panel samping, kontrol auto-labeling SAM, status bar, dan keyboard shortcut lengkap.

## Background

Spec terakhir yang mengintegrasikan semua modul: ImageManager (Spec 02), AnnotationCanvas (Spec 03), AutoLabeler (Spec 07), Visualizer (Spec 04), dan Exporter (Spec 05).

## Layout Aplikasi

```
┌──────────────────────────────────────────────────────────────────────┐
│ [Open Folder] [Save] | [✨Auto Label] [⚡Auto All] | [Draw][Select]  │ ← Toolbar
│ [Export COCO Splits] [Export XML] [Export Visualized] | [+][-][Fit]  │
├─────────────────────────────────────────┬────────────────────────────┤
│                                         │  📁 Dataset: 12 gambar     │
│                                         │  ████████░░ 8/12 anotasi  │
│                                         │  🤖 3 auto-labeled         │
│                                         ├────────────────────────────┤
│          ANNOTATION CANVAS              │  Mode: DRAW                │
│          (AnnotationCanvas)             │  ┌──────────────────────┐  │
│                                         │  │ SAM Controls         │  │
│                                         │  │ [Point] [Box] [Auto] │  │
│                                         │  │ Confidence: [0.75──] │  │
│                                         │  │ [✓ Accept All] [✗]   │  │
│                                         │  └──────────────────────┘  │
│                                         ├────────────────────────────┤
│                                         │  Instance List:            │
│                                         │  ● #1 manual               │
│                                         │  ● #2 SAM 0.92 ⚠️ review  │
│                                         ├────────────────────────────┤
│                                         │  [Thumbnail Preview]       │
├────────────────────────────────────────┬┴────────────────────────────┤
│ [◀] daun_001.jpg (1/12) [▶] | 2 inst. │ SAM_POINT | Zoom: 100%     │
└────────────────────────────────────────┴─────────────────────────────┘
```

## Module: `src/main_window.py`

### Class: `MainWindow(QMainWindow)`

```python
class MainWindow(QMainWindow):
    def __init__(self, config: Config): ...
    # File & navigasi
    def open_folder(self) -> None: ...
    def save_current(self) -> None: ...
    def next_image(self) -> None: ...
    def prev_image(self) -> None: ...
    # SAM auto-labeling
    def trigger_sam_point(self) -> None: ...      # switch ke SAM_POINT mode
    def trigger_sam_box(self) -> None: ...        # switch ke SAM_BOX mode
    def trigger_sam_auto_current(self) -> None:   # auto-label gambar aktif
    def trigger_sam_auto_all(self) -> None: ...   # auto-label semua gambar
    def accept_all_auto(self) -> None: ...
    def reject_selected_auto(self) -> None: ...
    # Export
    def export_coco_splits(self) -> None: ...
    def export_voc(self) -> None: ...
    def export_visualized(self) -> None: ...
    # Internal
    def _load_image_to_canvas(self, entry: ImageEntry) -> None: ...
    def _on_annotation_changed(self, polygons: list[Polygon]) -> None: ...
    def _on_point_clicked(self, x: float, y: float) -> None: ...   # forward ke AutoLabeler
    def _on_sam_result(self, polygons: list[Polygon]) -> None: ...  # inject ke canvas
    def _update_status_bar(self) -> None: ...
    def _update_instance_list(self, polygons: list[Polygon]) -> None: ...
    def _update_sam_controls(self) -> None: ...
```

## Toolbar

| Tombol | Aksi |
|--------|------|
| Open Folder | Buka dialog pilih folder dataset |
| Save | Simpan anotasi gambar aktif |
| ✨ Auto Label | Auto-label gambar aktif dengan SAM (full-image automatic) |
| ⚡ Auto All | Auto-label semua gambar yang belum dianotasi (batch, dengan progress dialog) |
| Draw | Switch ke DRAW mode |
| Select | Switch ke SELECT mode |
| Export COCO Splits | Export train/val/test JSON ke `output/coco/` |
| Export XML | Export Pascal VOC XML ke `output/voc/` |
| Export Visualized | Render + simpan semua gambar ke `output/visualized/` |
| Zoom+/−/Fit | Zoom canvas |

## Panel SAM Controls (Kanan Atas)

- **[Point]** — switch ke `SAM_POINT` mode, klik pada daun → SAM generate polygon
- **[Box]** — switch ke `SAM_BOX` mode, drag → SAM generate polygon dalam box
- **Confidence slider** — threshold filter untuk polygon SAM (0.50–0.99, default 0.75)
- **[✓ Accept All]** — konfirmasi semua polygon SAM pending
- **[✗ Reject All]** — tolak semua polygon SAM pending

## Instance List Panel (Kanan Tengah)

- Setiap row: warna swatch + `#id` + source label + (jika SAM: badge confidence + ikon ⚠️ "needs review")
- Klik row → select instance di canvas
- Tombol `×` di setiap row untuk hapus instance
- Row auto-scroll saat instance baru ditambah

## Auto Label All Dialog

Muncul saat `trigger_sam_auto_all()` dipanggil:
```
Auto-Label Semua Gambar

Akan memproses 4 gambar yang belum dianotasi.
Confidence threshold: [0.75──]
[☐ Skip gambar yang sudah dianotasi]

[Mulai]  [Batal]
```

Setelah selesai:
```
Auto-label selesai: 4 gambar diproses, 23 polygon digenerate.
Polygon menunggu review di setiap gambar sebelum export.
[OK]
```

## Keyboard Shortcuts

| Key | Aksi |
|-----|------|
| `N` | Gambar berikutnya (auto-save) |
| `P` | Gambar sebelumnya (auto-save) |
| `S` | Save anotasi aktif |
| `A` | Switch ke SAM_POINT mode |
| `B` | Switch ke SAM_BOX mode |
| `D` | Switch ke DRAW mode |
| `V` | Switch ke SELECT mode |
| `Enter` | Accept all pending SAM polygons |
| `Escape` | Batalkan polygon aktif / keluar SAM mode |
| `Delete` | Hapus instance yang dipilih |
| `Ctrl+Z` | Undo |
| `+` / `-` | Zoom in / out |
| `F` | Fit canvas |

## Auto-Save Behavior

- Saat pindah gambar: anotasi aktif otomatis disimpan (termasuk polygon SAM yang sudah di-accept)
- Polygon SAM yang masih pending (belum di-accept) **tidak ikut tersimpan**
- Notifikasi kecil muncul di status bar saat auto-save: `"Tersimpan ✓"`

## Acceptance Criteria

- [ ] `Open Folder` memuat gambar dan update progress bar
- [ ] Klik canvas di `SAM_POINT` mode memicu SAM dan inject polygon dengan border kuning
- [ ] `Accept All` memindahkan semua polygon pending ke list final dan update thumbnail
- [ ] `Auto All` menampilkan progress dialog dan berjalan di background thread (UI tidak freeze)
- [ ] Instance list menampilkan badge `SAM 0.92 ⚠️` untuk polygon yang perlu di-review
- [ ] `Export COCO Splits` menghasilkan 3 file JSON dan menampilkan ringkasan jumlah gambar per split
- [ ] Auto-save saat pindah gambar tidak menyimpan polygon SAM yang masih pending
- [ ] Semua keyboard shortcut berjalan sesuai tabel
- [ ] Status bar menampilkan info terkini termasuk mode SAM aktif

## Out of Scope

- Live training dari UI
- Import dari LabelMe / CVAT format eksternal
- Dark mode
