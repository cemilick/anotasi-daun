## ADDED Requirements

### Requirement: Export anotasi ke Pascal VOC XML per-gambar
Sistem SHALL mengekspor anotasi setiap gambar ke file XML terpisah di direktori output. Setiap file XML SHALL dapat di-parse oleh `xml.etree.ElementTree` tanpa error. Nama file XML SHALL sama dengan nama file gambar dengan ekstensi `.xml`.

#### Scenario: Satu file XML per gambar
- **WHEN** `export_voc_xml(image_manager, output_dir)` dipanggil dengan 5 gambar beranotasi
- **THEN** 5 file XML dihasilkan di `output_dir`, satu per gambar

#### Scenario: File XML dapat di-parse
- **WHEN** file XML dihasilkan
- **THEN** `xml.etree.ElementTree.parse(xml_path)` berhasil tanpa error

### Requirement: Struktur XML mengikuti Pascal VOC format
File XML SHALL memiliki elemen root `<annotation>` dengan child elements: `<folder>`, `<filename>`, `<path>`, `<source>`, `<size>`, `<segmented>`, dan satu atau lebih `<object>`.

#### Scenario: Elemen wajib ada di semua file XML
- **WHEN** file XML dihasilkan untuk gambar `daun_001.jpg`
- **THEN** XML memiliki `<filename>daun_001.jpg</filename>`, `<width>`, `<height>`, `<depth>3</depth>`, dan `<segmented>1</segmented>`

### Requirement: Setiap polygon menjadi satu elemen `<object>`
Setiap `Polygon` SHALL menjadi satu elemen `<object>` dalam XML dengan sub-elemen: `<name>daun_kelengkeh_itoh</name>`, `<bndbox>` (xmin, ymin, xmax, ymax), `<polygon>` (x1,y1,...), `<source>`, dan `<confidence>`.

#### Scenario: bndbox dari bounding box polygon
- **WHEN** polygon memiliki bounding box `[100, 180, 200, 260]`
- **THEN** XML memiliki `<xmin>100</xmin><ymin>180</ymin><xmax>200</xmax><ymax>260</ymax>`

#### Scenario: Koordinat polygon di dalam elemen polygon
- **WHEN** polygon memiliki 4 titik
- **THEN** XML memiliki `<x1>`, `<y1>`, `<x2>`, `<y2>`, dst. sebagai child dari `<polygon>`

### Requirement: Field source dan confidence tersimpan di XML
Setiap `<object>` SHALL menyertakan elemen `<source>` dan `<confidence>` dari data polygon.

#### Scenario: source dan confidence tersimpan
- **WHEN** polygon memiliki `source="sam_point"` dan `confidence=0.92`
- **THEN** `<object>` memiliki `<source>sam_point</source>` dan `<confidence>0.92</confidence>`

### Requirement: Gambar belum dianotasi dilewati
`export_voc_xml()` SHALL melewati gambar yang belum memiliki anotasi tanpa menghasilkan error dan tanpa membuat file XML kosong.

#### Scenario: Tidak ada file XML untuk gambar tanpa anotasi
- **WHEN** `ImageEntry` memiliki 0 polygon
- **THEN** tidak ada file XML yang dihasilkan untuk gambar tersebut

### Requirement: Export dengan filter split
`export_voc_xml()` SHALL menerima parameter `split` (`None`|`"train"`|`"val"`|`"test"`). Jika tidak `None`, hanya gambar dengan split yang sesuai yang diekspor.

#### Scenario: Filter split bekerja untuk VOC XML
- **WHEN** `export_voc_xml(image_manager, output_dir, split="val")` dipanggil
- **THEN** hanya file XML untuk gambar dengan `split="val"` yang dihasilkan
