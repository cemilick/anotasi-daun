# Spec 03 — Annotation Engine (Canvas Anotasi Interaktif)

## Goal

Widget PyQt5 canvas interaktif untuk menggambar polygon per-instance di atas gambar, mendukung multi-instance, edit titik, hapus, undo, dan menerima overlay polygon hasil auto-labeling dari SAM (Spec 07).

## Background

Pengguna klik-klik untuk membentuk polygon manual, atau mengkonfirmasi/mengedit polygon yang di-generate SAM secara otomatis. Setiap instance mendapat warna berbeda. Canvas harus bisa menerima polygon dari luar (inject dari AutoLabeler) dan menampilkannya tanpa mengganggu alur manual.

## Module: `src/annotation_engine.py`

### Class: `AnnotationCanvas(QWidget)`

```python
class AnnotationCanvas(QWidget):
    # Signals
    annotation_changed = pyqtSignal(list)   # emit list[Polygon] tiap ada perubahan
    status_message = pyqtSignal(str)        # emit pesan untuk status bar
    point_clicked = pyqtSignal(float, float) # emit koordinat klik untuk SAM point-prompt

    def __init__(self, config: Config, parent=None): ...
    def load_image(self, entry: ImageEntry, polygons: list[Polygon]) -> None: ...
    def get_polygons(self) -> list[Polygon]: ...
    def set_mode(self, mode: CanvasMode) -> None: ...
    def inject_polygons(self, polygons: list[Polygon]) -> None: ...  # dari AutoLabeler; memicu rekomputasi oklusi
    def accept_all_auto(self) -> None: ...    # konfirmasi semua polygon SAM
    def reject_auto_polygon(self, instance_id: int) -> None: ...
    def undo(self) -> None: ...
    def delete_selected(self) -> None: ...
    def clear_all(self) -> None: ...
    def zoom_in(self) -> None: ...
    def zoom_out(self) -> None: ...
    def reset_zoom(self) -> None: ...
```

### Enum: `CanvasMode`

```python
class CanvasMode(Enum):
    DRAW        = "draw"      # gambar polygon manual klik per titik
    SELECT      = "select"    # pilih & edit polygon existing
    SAM_POINT   = "sam_point" # klik 1 titik → SAM auto-generate polygon
    SAM_BOX     = "sam_box"   # drag kotak → SAM auto-generate polygon dalam area
    VIEW        = "view"      # read-only
```

### Enum: `OcclusionLevel`

```python
class OcclusionLevel(Enum):
    RENDAH  = "rendah"   # oklusi < 30%  — daun hampir sepenuhnya terlihat
    SEDANG  = "sedang"   # oklusi 30–60% — sebagian daun tertutup
    TINGGI  = "tinggi"   # oklusi > 60%  — sebagian besar daun tertutup
```

### Data Model: `Polygon`

```python
@dataclass
class Polygon:
    instance_id:      int                        # ID unik per instance dalam satu gambar
    points:           list[tuple[float, float]]  # koordinat pixel asli [(x, y), ...]
    class_id:         int                        # dari config.classes
    source:           str                        # "manual" | "sam_point" | "sam_auto"
    occlusion_level:  OcclusionLevel | None = None  # diisi otomatis; None = belum dihitung
    occlusion_ratio:  float | None = None           # 0.0–1.0; hasil komputasi mentah
    occlusion_manual: bool = False                  # True = user override, kunci dari rekomputasi
    is_confirmed:     bool = False               # False = pending SAM, True = final
```

> **Catatan**: `occlusion_level` diisi otomatis oleh `compute_occlusion_levels()`. Polygon dengan `occlusion_level = None` hanya terjadi sementara (antara polygon dibuat dan fungsi dipanggil); ditandai border oranye berkedip selama status ini berlangsung.

### Interaksi Mouse — Mode DRAW

| Aksi | Efek |
|------|------|
| Klik kiri | Tambah titik ke polygon aktif |
| Double-klik kiri | Tutup polygon (min 3 titik), siap instance baru |
| Klik kanan | Hapus titik terakhir |
| `Escape` | Batalkan polygon aktif |

### Interaksi Mouse — Mode SELECT

| Aksi | Efek |
|------|------|
| Klik polygon | Pilih instance (highlight border putih tebal) |
| Drag titik | Geser titik polygon yang dipilih |
| `Delete` | Hapus instance yang dipilih |
| Klik area kosong | Deselect |

### Interaksi Mouse — Mode SAM_POINT

| Aksi | Efek |
|------|------|
| Klik kiri | Emit `point_clicked(x, y)` → AutoLabeler proses → inject polygon hasil |
| Klik kanan | Tambah negative point (beri tahu SAM area yang BUKAN objek) |

### Interaksi Mouse — Mode SAM_BOX

| Aksi | Efek |
|------|------|
| Drag kiri | Gambar bounding box → AutoLabeler proses → inject polygon hasil |

### Oklusi per Instance

Tingkat oklusi ditentukan **otomatis** dengan `compute_occlusion_levels()` (dari `auto_labeler.py`) berdasarkan intersection area antar polygon. User dapat melihat dan meng-override hasilnya.

**Kapan komputasi otomatis dipicu:**

| Trigger | Aksi |
|---------|------|
| Polygon SAM di-inject via `inject_polygons()` | `compute_occlusion_levels()` dipanggil untuk seluruh set (existing + baru) |
| Polygon manual selesai ditutup (double-klik) | `compute_occlusion_levels()` dipanggil ulang untuk seluruh canvas |
| Polygon dihapus | `compute_occlusion_levels()` dipanggil ulang untuk sisa polygon |
| Polygon titik digeser di mode SELECT | `compute_occlusion_levels()` dipanggil ulang (geometri berubah) |

**Alur setelah polygon manual ditutup:**

1. Polygon baru ditambahkan ke canvas
2. `compute_occlusion_levels()` dipanggil → `occlusion_level` dan `occlusion_ratio` terisi otomatis
3. Toast kecil muncul di pojok canvas: `"Oklusi: Rendah (12%)"` (atau sesuai hasilnya) selama 2 detik
4. Polygon langsung dapat disimpan — tidak ada dialog konfirmasi wajib

**Override manual** (opsional, saat hasil otomatis dianggap tidak tepat):

Keyboard shortcut saat polygon dipilih di mode SELECT:
| Key | Efek |
|-----|------|
| `1` | Override oklusi → RENDAH (override manual ditandai dengan ikon kunci kecil `🔒` di badge) |
| `2` | Override oklusi → SEDANG |
| `3` | Override oklusi → TINGGI |
| `0` | Reset override → kembalikan ke hasil komputasi otomatis |

Polygon yang di-override manual **tidak** ikut diperbarui saat `compute_occlusion_levels()` dipanggil ulang (nilainya dikunci).

### Folder Output per Oklusi

Annotation JSON per-gambar disimpan di subfolder sesuai level oklusi **dominan** gambar (level yang paling banyak dimiliki instance dalam satu gambar):

```
annotations/
├── rendah/    # gambar dengan mayoritas instance oklusi rendah
├── sedang/    # gambar dengan mayoritas instance oklusi sedang
└── tinggi/    # gambar dengan mayoritas instance oklusi tinggi
```

Jika satu gambar memiliki instance dari beberapa level oklusi yang berbeda, gambar dikategorikan ke level **tertinggi** yang ada (tinggi > sedang > rendah) untuk memastikan tidak ada oklusi signifikan yang terlewat.

### Rendering

- Polygon **manual** (source=`"manual"`): fill warna solid semi-transparan, border solid 2px
- Polygon **auto SAM** yang belum dikonfirmasi (source=`"sam_point"` / `"sam_auto"`): fill semi-transparan + border **putus-putus kuning** + badge "SAM" kecil di pojok atas-kiri polygon
- Polygon yang **dipilih** (SELECT mode): border putih tebal 3px + drag handle kotak 6x6
- Polygon **oklusi belum dihitung** (`occlusion_level = None`): border oranye berkedip (blink interval 800ms)
- Label di centroid polygon: `#<id> [R 12%]` / `#<id> [S 45%]` / `#<id> [T 72%]` (badge level + persentase oklusi)
- Polygon dengan **override manual** (`occlusion_manual = True`): badge dengan ikon kunci, mis. `#<id> [R🔒]`
- **Loading indicator** saat SAM sedang memproses (spinner overlay di pojok canvas)

### Inject Polygons dari AutoLabeler

```
AutoLabeler.predict() → list[Polygon] dengan source = "sam_point" / "sam_auto"
→ canvas.inject_polygons(polygons)
→ tampilkan dengan border putus-putus kuning
→ user bisa accept (Enter / tombol Accept) atau reject (Delete)
→ setelah accept: source tetap, polygon masuk ke main list
```

### Zoom & Pan

- Zoom scroll mouse atau `+`/`-`
- Pan dengan middle-click drag atau Ctrl+drag
- `reset_zoom()`: fit-to-canvas

### Undo Stack

- Setiap aksi: tutup polygon, hapus, geser titik, inject, accept/reject → masuk undo stack
- Maksimal 30 langkah undo

## Acceptance Criteria

- [ ] Klik 3+ titik + double-klik menghasilkan polygon di `get_polygons()`
- [ ] `inject_polygons()` menampilkan polygon SAM dengan border kuning putus-putus
- [ ] Polygon SAM belum dikonfirmasi tidak masuk output `get_polygons()` sebelum di-accept
- [ ] `accept_all_auto()` memindahkan semua polygon pending ke list final
- [ ] Mode `SAM_POINT`: klik emit signal `point_clicked` dengan koordinat pixel asli (bukan canvas)
- [ ] Mode `SAM_BOX`: drag menghasilkan bounding box dalam koordinat pixel asli
- [ ] Loading spinner muncul saat SAM sedang diproses
- [ ] Undo membatalkan inject polygon SAM
- [ ] Zoom tidak mengubah koordinat polygon yang tersimpan
- [ ] `annotation_changed` dikirim hanya saat list final berubah (bukan saat pending)
- [ ] `occlusion_level` terisi otomatis setelah polygon di-inject atau ditutup manual (tanpa input user)
- [ ] Toast notifikasi muncul 2 detik setelah polygon manual ditutup, menampilkan level dan persentase oklusi
- [ ] `compute_occlusion_levels()` dipanggil ulang saat polygon ditambah, dihapus, atau titiknya digeser
- [ ] Polygon dengan `occlusion_manual = True` tidak berubah nilai oklusinya saat rekomputasi
- [ ] Shortcut `1`/`2`/`3` mengubah `occlusion_level` dan set `occlusion_manual = True`
- [ ] Shortcut `0` mereset override: `occlusion_manual = False` dan memicu rekomputasi
- [ ] Polygon dengan `occlusion_level = None` (transisi sesaat) ditampilkan dengan border oranye berkedip
- [ ] Label centroid menampilkan badge `[R 12%]`/`[S 45%]`/`[T 72%]` sesuai level dan rasio oklusi
- [ ] Polygon dengan override manual menampilkan badge `[R🔒]`/`[S🔒]`/`[T🔒]`
- [ ] `get_polygons()` hanya mengembalikan polygon dengan `occlusion_level != None`
- [ ] File JSON anotasi per-gambar disimpan di subfolder `annotations/rendah/`, `annotations/sedang/`, atau `annotations/tinggi/` sesuai level oklusi tertinggi dalam gambar

## Out of Scope

- SAM inference logic (Spec 07)
- Export (Spec 05)
- Bounding box annotation (hanya polygon)
