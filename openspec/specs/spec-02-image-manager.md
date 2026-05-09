# Spec 02 — Image Manager (Manajemen Dataset)

## Goal

Modul untuk memuat, menavigasi, melacak status anotasi, dan mengelola split dataset (train/val/test) untuk seluruh gambar dalam dataset.

## Background

Pengguna perlu bisa berpindah antar gambar, melihat mana yang sudah dianotasi (manual maupun auto-labeled), memuat ulang anotasi yang sudah tersimpan, dan akhirnya membagi dataset ke split train/val/test sebelum export ke COCO.

## Module: `src/image_manager.py`

### Class: `ImageManager`

```python
class ImageManager:
    def __init__(self, config: Config): ...
    def load_folder(self, path: str) -> int: ...          # return jumlah gambar
    def current_image(self) -> ImageEntry: ...
    def next(self) -> ImageEntry: ...
    def prev(self) -> ImageEntry: ...
    def go_to(self, index: int) -> ImageEntry: ...
    def get_all(self) -> list[ImageEntry]: ...
    def get_progress(self) -> AnnotationProgress: ...
    def save_annotation(self, entry: ImageEntry, polygons: list[Polygon]) -> None: ...
    def load_annotation(self, entry: ImageEntry) -> list[Polygon]: ...
    def is_annotated(self, entry: ImageEntry) -> bool: ...
    def generate_split(self) -> DatasetSplit: ...         # train/val/test split
    def get_split(self) -> DatasetSplit | None: ...       # load split yang sudah ada
```

### Dataclass: `ImageEntry`

```python
@dataclass
class ImageEntry:
    index: int
    filename: str
    filepath: str               # absolute path
    width: int
    height: int
    annotated: bool
    auto_labeled: bool          # True jika anotasi dihasilkan oleh SAM
    annotation_path: str        # path ke file .json di annotations/
    split: str | None           # "train" | "val" | "test" | None
```

### Dataclass: `Polygon`

```python
@dataclass
class Polygon:
    instance_id: int
    class_id: int
    class_name: str
    points: list[tuple[float, float]]   # koordinat (x, y) dalam pixel
    color: list[int]                    # [R, G, B]
    source: str                         # "manual" | "sam_auto" | "sam_point"
    confidence: float                   # 0.0–1.0 (1.0 untuk manual)
```

### Dataclass: `AnnotationProgress`

```python
@dataclass
class AnnotationProgress:
    total: int
    annotated: int             # manual + auto-labeled
    manual: int
    auto_labeled: int
    unreviewed_auto: int       # auto-labeled tapi belum dikonfirmasi user
```

### Dataclass: `DatasetSplit`

```python
@dataclass
class DatasetSplit:
    train: list[ImageEntry]
    val: list[ImageEntry]
    test: list[ImageEntry]
    seed: int
    created_at: str
```

### Format File Anotasi (`annotations/<nama_file>.json`)

```json
{
  "filename": "daun_001.jpg",
  "width": 3024,
  "height": 4032,
  "polygons": [
    {
      "instance_id": 1,
      "class_id": 1,
      "class_name": "daun_kelengkeh_itoh",
      "points": [[100, 200], [150, 180], [200, 220], [160, 260]],
      "color": [255, 82, 82],
      "source": "sam_point",
      "confidence": 0.92
    },
    {
      "instance_id": 2,
      "class_id": 1,
      "class_name": "daun_kelengkeh_itoh",
      "points": [[300, 400], [350, 380], [400, 420]],
      "color": [82, 255, 82],
      "source": "manual",
      "confidence": 1.0
    }
  ]
}
```

### File Split (`annotations/split.json`)

```json
{
  "seed": 42,
  "created_at": "2026-05-09",
  "train": ["daun_001.jpg", "daun_003.jpg"],
  "val":   ["daun_002.jpg"],
  "test":  ["daun_004.jpg"]
}
```

## Behavior

- `load_folder`: scan semua file dengan ekstensi dari config, sort by filename, baca dimensi via Pillow
- `save_annotation`: tulis JSON ke `annotations/<basename>.json`; update `entry.annotated = True`
- `load_annotation`: baca JSON; return list kosong jika file tidak ada
- `next` / `prev`: wrap-around (dari gambar terakhir → kembali ke pertama)
- `generate_split`: stratified split berdasarkan `config.dataset_split` (70/20/10), simpan ke `annotations/split.json`; jika split sudah ada, load dari file (tidak buat ulang)
- `get_progress`: hitung `unreviewed_auto` = jumlah gambar dengan source `sam_auto`/`sam_point` yang belum pernah di-save ulang oleh user

## Acceptance Criteria

- [ ] `load_folder` mengembalikan jumlah gambar yang benar dan mengisi `entry.width/height`
- [ ] Navigasi next/prev berjalan dengan wrap-around
- [ ] `save_annotation` menulis JSON valid dan `is_annotated` return `True` setelahnya
- [ ] `load_annotation` membaca ulang dengan benar setelah restart aplikasi
- [ ] Polygon dengan `source = "sam_auto"` tersimpan dan terbaca dengan benar
- [ ] `generate_split` menghasilkan split 70/20/10 tanpa overlap antar split
- [ ] Memanggil `generate_split` kedua kali mengembalikan split yang sama (load dari file)
- [ ] `get_progress.unreviewed_auto` akurat

## Out of Scope

- UI navigasi (Spec 06)
- Rendering canvas (Spec 03)
- Auto-labeling logic (Spec 07)
