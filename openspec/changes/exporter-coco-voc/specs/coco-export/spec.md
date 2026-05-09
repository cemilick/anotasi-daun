## ADDED Requirements

### Requirement: Export anotasi ke COCO JSON dengan RLE mask
Sistem SHALL mengekspor seluruh anotasi polygon dari `ImageManager` ke file COCO JSON yang dapat dibaca oleh `pycocotools.coco.COCO(path)`. Field `segmentation` SHALL berupa RLE dict dengan key `"counts"` (string, hasil `decode("utf-8")`) dan `"size"` (`[height, width]`). Field `segmentation_polygon` SHALL disertakan sebagai fallback flat list `[x1,y1,x2,y2,...]`.

#### Scenario: Export menghasilkan JSON yang valid untuk pycocotools
- **WHEN** `export_coco(image_manager, output_path)` dipanggil dengan anotasi yang ada
- **THEN** file JSON yang dihasilkan dapat dibaca dengan `pycocotools.coco.COCO(output_path)` tanpa error

#### Scenario: Segmentation berformat RLE dict
- **WHEN** annotation di-export dengan `use_rle=True`
- **THEN** setiap annotation memiliki field `segmentation` berupa dict dengan key `"counts"` (str) dan `"size"` ([int, int])

#### Scenario: Polygon fallback selalu disertakan
- **WHEN** annotation di-export (baik `use_rle=True` maupun `False`)
- **THEN** setiap annotation memiliki field `segmentation_polygon` berupa list flat koordinat `[x1,y1,x2,y2,...]`

### Requirement: Bbox format COCO [x_min, y_min, width, height]
Field `bbox` pada setiap annotation COCO SHALL berformat `[x_min, y_min, width, height]` — bukan `[x_min, y_min, x_max, y_max]`.

#### Scenario: Bbox dihitung dari bounding box polygon
- **WHEN** polygon memiliki koordinat `[(100,200),(150,180),(200,220),(160,260)]`
- **THEN** `bbox` adalah `[100, 180, 100, 80]` (x_min=100, y_min=180, width=100, height=80)

### Requirement: Area dihitung dengan Shoelace formula
Field `area` pada setiap annotation COCO SHALL dihitung menggunakan Shoelace formula dari koordinat polygon, bukan dari bbox.

#### Scenario: Area Shoelace lebih akurat dari bbox area
- **WHEN** polygon non-rectangular di-export
- **THEN** `area` pada annotation lebih kecil dari `bbox[2] * bbox[3]` (luas bbox)

### Requirement: Metadata anotasi tersimpan di COCO JSON
Setiap annotation COCO SHALL menyertakan field: `source` (str), `confidence` (float), `occlusion_level` (str|null), `occlusion_ratio` (float|null).

#### Scenario: Field source dan confidence tersimpan
- **WHEN** polygon memiliki `source="sam_point"` dan `confidence=0.92`
- **THEN** annotation COCO memiliki `"source": "sam_point"` dan `"confidence": 0.92`

#### Scenario: occlusion_level null jika belum diset
- **WHEN** polygon memiliki `occlusion_level=None`
- **THEN** annotation COCO memiliki `"occlusion_level": null`

### Requirement: Annotation ID unik secara global
`id` pada setiap annotation SHALL berupa integer yang unik secara global di seluruh file JSON, auto-increment mulai dari 1.

#### Scenario: Tidak ada annotation ID duplikat
- **WHEN** dataset memiliki 10 gambar dengan masing-masing 3 polygon
- **THEN** annotation IDs adalah 1 sampai 30 tanpa duplikat

### Requirement: Gambar belum dianotasi dilewati
`export_coco()` SHALL melewati gambar yang belum memiliki anotasi tanpa menghasilkan error.

#### Scenario: Gambar tanpa polygon dilewati
- **WHEN** `ImageEntry` memiliki 0 polygon
- **THEN** gambar tersebut tidak muncul di `"images"` maupun `"annotations"` pada output JSON

### Requirement: Export dengan filter split
`export_coco()` SHALL menerima parameter `split` (`None`|`"train"`|`"val"`|`"test"`). Jika `split` tidak `None`, hanya gambar dengan `entry.split == split` yang diekspor.

#### Scenario: Filter split train menghasilkan hanya gambar train
- **WHEN** `export_coco(image_manager, path, split="train")` dipanggil
- **THEN** hanya gambar dengan `split="train"` yang muncul di output JSON

#### Scenario: Split None mengekspor semua gambar
- **WHEN** `export_coco(image_manager, path, split=None)` dipanggil
- **THEN** semua gambar yang memiliki anotasi diekspor tanpa filter split
