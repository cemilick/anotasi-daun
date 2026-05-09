# Spec 04 — Visualizer (Overlay Mask Berwarna)

## Goal

Modul untuk menghasilkan gambar visualisasi hasil anotasi: foto asli + overlay mask berwarna per-instance, bisa ditampilkan di UI (QPixmap) maupun disimpan ke disk. Mendukung visualisasi perbedaan antara polygon manual dan auto-labeled SAM.

## Background

Setelah anotasi selesai, pengguna ingin melihat dan menyimpan hasilnya sebagai gambar dengan setiap daun diwarnai berbeda. Ini juga digunakan untuk quality check sebelum training: apakah mask hasil SAM sudah rapi atau perlu koreksi manual.

## Module: `src/visualizer.py`

### Class: `Visualizer`

```python
class Visualizer:
    def __init__(self, config: Config): ...

    def render(
        self,
        image_path: str,
        polygons: list[Polygon],
        show_labels: bool = True,
        show_border: bool = True,
        highlight_auto: bool = False,  # beri tanda visual pada polygon SAM
    ) -> np.ndarray:
        """Return gambar BGR numpy array dengan overlay mask."""
        ...

    def save(
        self,
        image_path: str,
        polygons: list[Polygon],
        output_path: str,
        show_labels: bool = True,
        highlight_auto: bool = False,
    ) -> str:
        """Render dan simpan ke output_path. Return path file."""
        ...

    def render_thumbnail(
        self,
        image_path: str,
        polygons: list[Polygon],
        max_size: tuple[int, int] = (300, 300),
    ) -> QPixmap:
        """Return QPixmap thumbnail untuk panel samping UI."""
        ...

    def render_comparison(
        self,
        image_path: str,
        polygons_before: list[Polygon],
        polygons_after: list[Polygon],
    ) -> np.ndarray:
        """
        Side-by-side: kiri = SAM raw, kanan = setelah koreksi manual.
        Berguna untuk QC dataset.
        """
        ...

    def export_all(
        self,
        image_manager: ImageManager,
        output_dir: str,
        progress_callback: callable = None,
    ) -> list[str]:
        """Render dan simpan semua gambar yang sudah dianotasi."""
        ...
```

### Spesifikasi Visual

#### Mask Per-Instance
- Setiap polygon diisi warna dari `polygon.color` dengan opacity `config.canvas.mask_opacity` (0.45)
- Blend: `output = image * (1 - alpha) + mask_color * alpha`

#### Polygon Auto-Labeled (saat `highlight_auto=True`)
- Border warna kuning `[255, 220, 0]`, tebal 2px, **putus-putus** (draw setiap 8px)
- Pojok kiri atas: label kecil `SAM` + confidence score (misal: `SAM 0.92`)

#### Polygon Manual
- Border solid warna yang sama dengan mask, tebal 2px

#### Label Instance
- Teks `#<id>` di centroid polygon
- Background: kotak putih semi-transparan
- Font: `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.5

#### Nama File Output
```
output/visualized/daun_001_annotated.jpg
```

### Algoritma Render

```
1. Baca gambar asli dengan OpenCV (BGR)
2. Buat layer overlay (copy gambar asli)
3. Untuk setiap polygon dalam urutan instance_id:
   a. Convert points ke numpy int32
   b. cv2.fillPoly(overlay, [points], color)
4. cv2.addWeighted(overlay, alpha, image, 1-alpha, 0) → blended
5. Untuk setiap polygon:
   a. Gambar border (solid atau putus-putus sesuai source)
   b. Tulis label di centroid
   c. Jika highlight_auto dan source != "manual": tambahkan badge "SAM"
6. Return blended
```

## Acceptance Criteria

- [ ] `render()` menghasilkan gambar dengan setiap polygon terisi warna semi-transparan
- [ ] Foto asli masih terlihat di balik mask (tidak tertutup penuh)
- [ ] `highlight_auto=True`: polygon SAM menampilkan border kuning putus-putus + badge confidence
- [ ] `render_thumbnail()` mengembalikan QPixmap valid yang bisa ditampilkan di PyQt5
- [ ] `render_comparison()` menghasilkan gambar side-by-side lebar 2x gambar asli
- [ ] `export_all()` memanggil `progress_callback(current, total)` tiap gambar selesai
- [ ] Gambar tanpa anotasi mengembalikan gambar asli tanpa modifikasi
- [ ] File output di-save sebagai JPG kualitas 95

## Out of Scope

- Semantic segmentation (satu warna untuk semua daun)
- Video / GIF animasi
- Format output selain JPG/PNG
