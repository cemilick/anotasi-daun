## ADDED Requirements

### Requirement: Config singleton tersedia global
Sistem SHALL menyediakan class `Config` dengan classmethod `get()` yang mengembalikan instance singleton. Instance dibuat satu kali (lazy) saat pertama dipanggil dan digunakan kembali untuk semua pemanggilan berikutnya.

#### Scenario: Pemanggilan pertama membuat instance
- **WHEN** `Config.get()` dipanggil untuk pertama kali
- **THEN** sistem membuat instance baru dengan memuat `config.json` dari direktori root proyek

#### Scenario: Pemanggilan berikutnya mengembalikan instance yang sama
- **WHEN** `Config.get()` dipanggil lebih dari satu kali
- **THEN** sistem mengembalikan instance yang identik tanpa membaca ulang file

### Requirement: Config memuat config.json dari root proyek
Sistem SHALL memuat `config.json` dari direktori root proyek. Jika file tidak ditemukan, SHALL raise `FileNotFoundError` dengan pesan yang menyertakan path absolut yang dicari.

#### Scenario: File config.json ditemukan
- **WHEN** `config.json` ada di direktori root
- **THEN** sistem memuat seluruh isi JSON ke dalam atribut Config yang dapat diakses secara hierarkis

#### Scenario: File config.json tidak ditemukan
- **WHEN** `config.json` tidak ada di direktori root
- **THEN** sistem raise `FileNotFoundError` dengan pesan yang menyertakan path absolut yang dicari

### Requirement: Akses config secara hierarkis via dot-notation
Sistem SHALL memungkinkan akses ke nilai konfigurasi dengan dot-notation (misal: `Config.get().sam.device`, `Config.get().paths.images`).

#### Scenario: Akses nilai nested
- **WHEN** kode memanggil `Config.get().sam.points_per_side`
- **THEN** sistem mengembalikan nilai integer yang sesuai dari `config.json["sam"]["points_per_side"]`

### Requirement: Semua path config sebagai pathlib.Path
Sistem SHALL mengekspos semua nilai dalam `config.paths` sebagai objek `pathlib.Path` relatif terhadap direktori root proyek.

#### Scenario: Akses path images
- **WHEN** kode memanggil `Config.get().paths.images`
- **THEN** sistem mengembalikan `pathlib.Path("images")` atau path absolut yang ekuivalen

#### Scenario: Akses path subfolder oklusi
- **WHEN** kode memanggil `Config.get().paths.annotations_rendah`
- **THEN** sistem mengembalikan `pathlib.Path("annotations/rendah")`

### Requirement: Auto-fallback device CUDA ke CPU
Sistem SHALL secara otomatis mengubah nilai `sam.device` dari `"cuda"` ke `"cpu"` dalam instance (tidak mengubah file `config.json`) apabila CUDA tidak tersedia di runtime.

#### Scenario: CUDA tersedia
- **WHEN** `config.json` menyetel `sam.device = "cuda"` dan `torch.cuda.is_available()` mengembalikan `True`
- **THEN** `Config.get().sam.device` mengembalikan `"cuda"`

#### Scenario: CUDA tidak tersedia
- **WHEN** `config.json` menyetel `sam.device = "cuda"` dan `torch.cuda.is_available()` mengembalikan `False`
- **THEN** `Config.get().sam.device` mengembalikan `"cpu"` tanpa error dan tanpa memodifikasi `config.json`

#### Scenario: Torch tidak terinstall
- **WHEN** paket `torch` tidak terinstall dan `config.sam.device = "cuda"`
- **THEN** sistem tidak raise `ImportError`; `Config.get().sam.device` mengembalikan `"cpu"`

### Requirement: Override device ke CPU via force_cpu()
Sistem SHALL menyediakan classmethod `Config.force_cpu()` yang mengubah `sam.device` ke `"cpu"` dalam instance aktif, dapat dipanggil setelah `Config.get()` dan sebelum modul lain menggunakan config.

#### Scenario: Paksa CPU via method
- **WHEN** `Config.force_cpu()` dipanggil setelah `Config.get()`
- **THEN** `Config.get().sam.device` mengembalikan `"cpu"` untuk sisa sesi, tanpa memodifikasi file
