## Context

Proyek ini adalah aplikasi labeling daun (daun_kelengkeh_itoh) berbasis Python. Belum ada modul pengelolaan dataset — navigasi gambar dan persistensi anotasi masih belum diimplementasikan. Modul ini menjadi tulang punggung state management: semua komponen lain (UI, annotation engine, exporter) bergantung pada `ImageManager` untuk mengetahui gambar mana yang aktif dan anotasi apa yang sudah tersimpan.

## Goals / Non-Goals

**Goals:**
- Muat seluruh gambar dari folder sekaligus (termasuk dimensi via Pillow)
- Navigasi O(1) via indeks dengan wrap-around
- Persistensi anotasi per gambar sebagai JSON di `annotations/`
- Generate dan cache dataset split 70/20/10 yang reproducible
- Track `unreviewed_auto`: anotasi SAM yang belum pernah di-save ulang oleh user

**Non-Goals:**
- Rendering canvas atau overlay anotasi (spec-03)
- Logika auto-labeling dengan SAM (spec-07)
- UI navigasi dan panel kontrol (spec-06)
- Export ke format COCO (spec-05)

## Decisions

### 1. Pillow untuk membaca dimensi gambar saat `load_folder`

**Pilihan**: Baca semua dimensi saat load, simpan di `ImageEntry`.
**Alternatif**: Baca on-demand saat gambar dibuka.
**Alasan**: Pillow sudah ada sebagai dependensi proyek. Membaca dimensi saat load memungkinkan `get_progress` dan `generate_split` bekerja tanpa harus membuka file gambar lagi. Tradeoff: startup lebih lambat untuk folder besar, tapi konsistensi lebih baik.

### 2. Satu file JSON per gambar di `annotations/`

**Pilihan**: `annotations/<basename>.json` (tanpa ekstensi gambar) per gambar.
**Alternatif**: SQLite database atau satu file JSON besar.
**Alasan**: File per gambar memudahkan debugging, memungkinkan partial save tanpa korupsi seluruh dataset, dan kompatibel dengan git-based version control. Tidak perlu dependensi eksternal.

### 3. Deteksi `unreviewed_auto` berbasis flag `auto_labeled` di `ImageEntry`

**Pilihan**: Set `entry.auto_labeled = True` saat load annotation jika semua polygon bersumber dari SAM. Set `False` saat user menyimpan ulang (via `save_annotation`).
**Alternatif**: Timestamp-based comparison antara file anotasi dan last-save.
**Alasan**: Flag lebih sederhana dan tidak bergantung pada filesystem timestamps yang bisa berubah karena copy/sync. `auto_labeled` diset ke `False` setelah user explicitly menyimpan, menandai anotasi telah direview.

### 4. Reproducible split dengan `random.seed`

**Pilihan**: Shuffle list gambar dengan seed tetap (default 42 dari config), simpan hasil ke `split.json`.
**Alternatif**: Stratified split berdasarkan metadata kelas.
**Alasan**: Dataset ini hanya satu kelas (`daun_kelengkeh_itoh`), sehingga stratified split tidak memberikan nilai tambah. Seed yang tersimpan di `split.json` menjamin reproducibility saat load ulang. Memanggil `generate_split` kedua kali akan load dari file, bukan generate ulang.

### 5. Wrap-around navigasi

**Pilihan**: `next()` dari gambar terakhir kembali ke indeks 0; `prev()` dari gambar pertama kembali ke indeks terakhir.
**Alasan**: UX standar untuk image viewer — mencegah user "mentok" di ujung daftar.

## Risks / Trade-offs

- **Startup lambat untuk dataset besar** → Pillow membaca header gambar (bukan seluruh pixel), sehingga dampaknya minimal untuk dataset ratusan gambar. Jika lebih dari 10k gambar, pertimbangkan lazy loading.
- **File anotasi tidak sinkron dengan gambar** (gambar dihapus tapi JSON masih ada) → `load_folder` tidak membersihkan orphan JSON; ini di luar scope modul ini.
- **Concurrent write** (dua proses menulis anotasi bersamaan) → Aplikasi ini single-process, sehingga ini tidak relevan.
