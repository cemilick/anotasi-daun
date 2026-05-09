## MODIFIED Requirements

### Requirement: Annotation persistence
`ImageManager` SHALL save a list of `Polygon` objects for a given `ImageEntry` to the correct occlusion subfolder (`annotations/rendah/`, `annotations/sedang/`, or `annotations/tinggi/`) based on the highest `occlusion_level` present among the polygon list, and load them back correctly. If no polygon has a computed `occlusion_level`, the file SHALL be saved to `annotations/rendah/` as a fallback. After `save_annotation`, `is_annotated(entry)` SHALL return `True`. `load_annotation` SHALL return an empty list if no annotation file exists. `load_annotation` SHALL search all three subfolders when locating an existing annotation file.

#### Scenario: Save with SEDANG and TINGGI polygons goes to tinggi subfolder
- **WHEN** `save_annotation(entry, polygons)` is called where polygons include both `OcclusionLevel.SEDANG` and `OcclusionLevel.TINGGI` instances
- **THEN** the JSON file is written to `annotations/tinggi/<basename>.json`

#### Scenario: Save with all RENDAH polygons goes to rendah subfolder
- **WHEN** `save_annotation(entry, polygons)` is called where all polygons have `occlusion_level = OcclusionLevel.RENDAH`
- **THEN** the JSON file is written to `annotations/rendah/<basename>.json`

#### Scenario: Load after restart finds file in subfolder
- **WHEN** `load_annotation(entry)` is called after application restart for an image whose annotation was saved in `annotations/sedang/`
- **THEN** it returns the correct polygon list from that subfolder

#### Scenario: Load non-existent annotation
- **WHEN** `load_annotation(entry)` is called for an image with no annotation file in any subfolder
- **THEN** it returns an empty list

#### Scenario: Polygon round-trip preserves occlusion fields
- **WHEN** a polygon with `occlusion_level = OcclusionLevel.SEDANG`, `occlusion_ratio = 0.45`, `occlusion_manual = True`, and `is_confirmed = True` is saved and loaded
- **THEN** all four occlusion fields are preserved exactly
