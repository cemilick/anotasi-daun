## 1. Setup dan Dependensi

- [x] 1.1 Tambah `pycocotools` ke `requirements.txt` (Phase 1 dependency)
- [x] 1.2 Buat file `src/exporter.py` dengan class `Exporter` dan method stubs

## 2. Utilitas Konversi

- [x] 2.1 Implementasi fungsi `polygon_to_rle(points, height, width)` menggunakan `cv2.fillPoly` dan `pycocotools.mask.encode`
- [x] 2.2 Implementasi fungsi `compute_area_shoelace(points)` dari koordinat polygon
- [x] 2.3 Implementasi fungsi `compute_bbox(points)` yang mengembalikan `[x_min, y_min, width, height]`

## 3. Export COCO JSON

- [x] 3.1 Implementasi `export_single_coco(entry, polygons, annotation_id_start)` — return dict COCO satu gambar tanpa tulis file
- [x] 3.2 Implementasi `export_coco(image_manager, output_path, split, use_rle, progress_callback)` dengan filter split dan global annotation ID counter
- [x] 3.3 Pastikan gambar tanpa anotasi dilewati (guard kondisi `len(polygons) == 0`)
- [x] 3.4 Pastikan `segmentation_polygon` selalu disertakan sebagai fallback di setiap annotation
- [x] 3.5 Pastikan field `occlusion_level` dan `occlusion_ratio` ada di setiap annotation (`null` jika belum diset)
- [x] 3.6 Implementasi `export_coco_splits(image_manager, output_dir, progress_callback)` — hasilkan `train.json`, `val.json`, `test.json`

## 4. Export Pascal VOC XML

- [x] 4.1 Implementasi `export_voc_xml(image_manager, output_dir, split, progress_callback)` dengan struktur XML sesuai format Pascal VOC
- [x] 4.2 Pastikan setiap `<object>` menyertakan `<bndbox>` (xmin/ymin/xmax/ymax), `<polygon>` (x1,y1,...), `<source>`, dan `<confidence>`
- [x] 4.3 Pastikan gambar tanpa anotasi tidak menghasilkan file XML

## 5. Validasi dan Testing

- [x] 5.1 Tulis test: output `export_coco()` dapat dibaca oleh `pycocotools.coco.COCO(path)` tanpa error
- [x] 5.2 Tulis test: tidak ada overlap image ID antar tiga file split dari `export_coco_splits()`
- [x] 5.3 Tulis test: output `export_voc_xml()` dapat di-parse oleh `xml.etree.ElementTree.parse()`
- [x] 5.4 Tulis test: area dari `compute_area_shoelace()` lebih kecil dari `bbox_width * bbox_height` untuk polygon non-rectangular
