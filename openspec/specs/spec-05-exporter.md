# Spec 05 — Exporter (COCO JSON, Pascal VOC XML, Dataset Split)

## Goal

Modul untuk mengekspor seluruh anotasi ke format COCO JSON (dengan RLE mask, siap untuk Mask-RCNN) dan Pascal VOC XML, dengan pembagian train/val/test split otomatis.

## Background

Output COCO JSON dari exporter ini adalah **antarmuka antara Phase 1 (auto-labeling) dan Phase 2 (training)**. Spec 08 (Mask-RCNN + ASPP) mengonsumsi file-file ini sebagai satu-satunya input dari Phase 1. Format harus kompatibel dengan `pycocotools` dan `torchvision.datasets.CocoDetection`.

`pycocotools` adalah dependensi **Phase 1** (`requirements.txt`) karena dibutuhkan di sini untuk RLE encoding — bukan hanya untuk evaluasi training.

## Module: `src/exporter.py`

### Class: `Exporter`

```python
class Exporter:
    def __init__(self, config: Config): ...

    def export_coco(
        self,
        image_manager: ImageManager,
        output_path: str,
        split: str | None = None,      # None = all, "train"/"val"/"test"
        use_rle: bool = True,           # True = RLE mask, False = polygon segmentation
        progress_callback: callable = None,
    ) -> str:
        """
        Export ke COCO JSON. Return path file output.
        """
        ...

    def export_coco_splits(
        self,
        image_manager: ImageManager,
        output_dir: str,
        progress_callback: callable = None,
    ) -> dict[str, str]:
        """
        Export train.json, val.json, test.json sekaligus.
        Return {"train": path, "val": path, "test": path}.
        """
        ...

    def export_voc_xml(
        self,
        image_manager: ImageManager,
        output_dir: str,
        split: str | None = None,
        progress_callback: callable = None,
    ) -> list[str]:
        """Export tiap gambar ke file XML terpisah (Pascal VOC). Return list path."""
        ...

    def export_single_coco(
        self,
        entry: ImageEntry,
        polygons: list[Polygon],
        annotation_id_start: int = 1,
    ) -> dict:
        """Return dict COCO untuk satu gambar (untuk preview/debug)."""
        ...
```

---

## Format COCO JSON

### Struktur Output: `output/coco/annotations.json` (atau `train.json`, dll.)

```json
{
  "info": {
    "description": "Daun Kelengkeh Itoh Dataset",
    "version": "1.0",
    "year": 2026,
    "contributor": "",
    "date_created": "2026-05-09",
    "split": "train"
  },
  "licenses": [],
  "categories": [
    {
      "id": 1,
      "name": "daun_kelengkeh_itoh",
      "supercategory": "plant"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "daun_001.jpg",
      "width": 3024,
      "height": 4032,
      "date_captured": "",
      "split": "train"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": {
        "counts": "...",
        "size": [4032, 3024]
      },
      "segmentation_polygon": [[100, 200, 150, 180, 200, 220, 160, 260]],
      "bbox": [100, 180, 100, 80],
      "area": 3200.0,
      "iscrowd": 0,
      "source": "sam_point",
      "confidence": 0.92,
      "occlusion_level": "sedang",
      "occlusion_ratio": 0.43
    }
  ]
}
```

### Aturan Konversi

| Field | Aturan |
|-------|--------|
| `segmentation` | RLE format via `pycocotools.mask.encode(np.asfortranarray(binary_mask))` |
| `segmentation_polygon` | Tetap disimpan sebagai fallback: flat list `[x1,y1,x2,y2,...]` |
| `bbox` | `[x_min, y_min, width, height]` dari bounding box polygon |
| `area` | Shoelace formula dari koordinat polygon |
| `iscrowd` | Selalu `0` (instance segmentation) |
| `source` | Dari `polygon.source` ("manual" / "sam_point" / "sam_auto") |
| `confidence` | Dari `polygon.confidence` |
| `occlusion_level` | Dari `polygon.occlusion_level.value` ("rendah" / "sedang" / "tinggi") |
| `occlusion_ratio` | Dari `polygon.occlusion_ratio` (float 0.0–1.0), `null` jika belum dihitung |
| `id` annotation | Auto-increment global unik di seluruh dataset |

### Binary Mask dari Polygon

```python
# Konversi polygon ke binary mask untuk RLE
from pycocotools import mask as coco_mask
import numpy as np

def polygon_to_rle(points, height, width):
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(binary_mask, [pts], 1)
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")  # JSON serializable
    return rle
```

---

## Format Pascal VOC XML

### Struktur Output: `output/voc/daun_001.xml`

```xml
<annotation>
  <folder>images</folder>
  <filename>daun_001.jpg</filename>
  <path>/absolute/path/images/daun_001.jpg</path>
  <source>
    <database>Daun Kelengkeh Itoh Dataset</database>
  </source>
  <size>
    <width>3024</width>
    <height>4032</height>
    <depth>3</depth>
  </size>
  <segmented>1</segmented>
  <object>
    <name>daun_kelengkeh_itoh</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <source>sam_point</source>
    <confidence>0.92</confidence>
    <bndbox>
      <xmin>100</xmin>
      <ymin>180</ymin>
      <xmax>200</xmax>
      <ymax>260</ymax>
    </bndbox>
    <polygon>
      <x1>100</x1><y1>200</y1>
      <x2>150</x2><y2>180</y2>
      <x3>200</x3><y3>220</y3>
      <x4>160</x4><y4>260</y4>
    </polygon>
  </object>
</annotation>
```

---

## Acceptance Criteria

- [ ] `export_coco()` menghasilkan JSON yang dapat dibaca `pycocotools.coco.COCO(path)`
- [ ] `segmentation` berupa RLE dict dengan key `"counts"` (string) dan `"size"`
- [ ] `segmentation_polygon` disertakan sebagai fallback di setiap annotation
- [ ] `bbox` format COCO: `[x_min, y_min, width, height]`
- [ ] `area` dihitung dengan Shoelace formula (bukan perkiraan bbox)
- [ ] `export_coco_splits()` menghasilkan 3 file: `train.json`, `val.json`, `test.json`
- [ ] Tidak ada overlap image antara train/val/test dalam split export
- [ ] `export_voc_xml()` menghasilkan XML valid yang dapat di-parse `xml.etree.ElementTree`
- [ ] Gambar belum dianotasi dilewati tanpa error
- [ ] Field `source` dan `confidence` tersimpan di JSON dan XML
- [ ] Field `occlusion_level` dan `occlusion_ratio` tersimpan di setiap annotation COCO JSON
- [ ] Polygon dengan `occlusion_level = None` diekspor dengan `"occlusion_level": null`

## Out of Scope

- Export ke YOLO TXT format
- Upload ke platform anotasi eksternal (CVAT cloud, Roboflow, dsb.)
- Augmentasi data (Spec 08)
