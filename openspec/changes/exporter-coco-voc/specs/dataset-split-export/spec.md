## ADDED Requirements

### Requirement: Export tiga file split sekaligus
Sistem SHALL menyediakan method `export_coco_splits()` yang mengekspor tiga file JSON sekaligus: `train.json`, `val.json`, dan `test.json` ke direktori yang ditentukan. Method SHALL mengembalikan dict `{"train": path, "val": path, "test": path}`.

#### Scenario: Tiga file dihasilkan sekaligus
- **WHEN** `export_coco_splits(image_manager, output_dir)` dipanggil
- **THEN** tiga file `train.json`, `val.json`, `test.json` dihasilkan di `output_dir`

#### Scenario: Return value berisi path ke tiga file
- **WHEN** `export_coco_splits()` selesai
- **THEN** return value adalah dict dengan key `"train"`, `"val"`, `"test"` yang masing-masing berisi path absolut file JSON yang dihasilkan

### Requirement: Tidak ada overlap image antar split
Setiap gambar SHALL muncul di paling banyak satu dari `train.json`, `val.json`, atau `test.json`. Overlap antar-split dilarang.

#### Scenario: Image IDs tidak duplikat antar file split
- **WHEN** ketiga file split dihasilkan dan image IDs dari masing-masing file dikumpulkan
- **THEN** tidak ada image ID yang muncul di lebih dari satu file

### Requirement: Filter split menggunakan properti ImageEntry
`export_coco_splits()` SHALL menentukan file tujuan setiap gambar berdasarkan nilai `entry.split` dari `ImageManager`. Gambar dengan `entry.split == "train"` masuk ke `train.json`, dst.

#### Scenario: Gambar difilter berdasarkan entry.split
- **WHEN** dataset memiliki 7 gambar train, 2 gambar val, 1 gambar test
- **THEN** `train.json` berisi 7 gambar, `val.json` berisi 2, `test.json` berisi 1

### Requirement: File split yang kosong tetap dihasilkan
`export_coco_splits()` SHALL menghasilkan file JSON yang valid meski tidak ada gambar untuk split tertentu (daftar `"images"` dan `"annotations"` kosong).

#### Scenario: Split dengan 0 gambar menghasilkan JSON kosong yang valid
- **WHEN** tidak ada gambar dengan `split="test"`
- **THEN** `test.json` tetap dihasilkan dengan `"images": []` dan `"annotations": []`

### Requirement: Preview export satu gambar via export_single_coco
Sistem SHALL menyediakan method `export_single_coco(entry, polygons, annotation_id_start)` yang mengembalikan dict COCO untuk satu gambar tanpa menulis ke file. Digunakan untuk preview dan debugging.

#### Scenario: export_single_coco mengembalikan dict tanpa menulis file
- **WHEN** `export_single_coco(entry, polygons)` dipanggil
- **THEN** method mengembalikan dict Python dengan key `"images"` dan `"annotations"` tanpa membuat file apapun
