## ADDED Requirements

### Requirement: Load folder
`ImageManager` SHALL scan a given directory for image files with extensions defined in `Config`, sort them by filename, read each image's dimensions via Pillow, and return the total image count. Each image is represented as an `ImageEntry` with `index`, `filename`, `filepath`, `width`, `height`, `annotated`, `auto_labeled`, `annotation_path`, and `split` fields.

#### Scenario: Folder with images
- **WHEN** `load_folder(path)` is called on a directory containing 5 JPEG files
- **THEN** it returns `5`, and each `ImageEntry` has correct `width` and `height` values read from the file

#### Scenario: Folder with no matching files
- **WHEN** `load_folder(path)` is called on a directory with no files matching configured extensions
- **THEN** it returns `0` and the internal image list is empty

### Requirement: Navigation with wrap-around
`ImageManager` SHALL support `next()`, `prev()`, and `go_to(index)` methods that return the corresponding `ImageEntry`. `next()` from the last image SHALL wrap to index 0. `prev()` from index 0 SHALL wrap to the last image.

#### Scenario: Next from last image
- **WHEN** the current image is the last one and `next()` is called
- **THEN** the returned `ImageEntry` has `index == 0`

#### Scenario: Prev from first image
- **WHEN** the current image is index 0 and `prev()` is called
- **THEN** the returned `ImageEntry` has `index` equal to the total count minus 1

#### Scenario: Go to specific index
- **WHEN** `go_to(3)` is called on a dataset with at least 4 images
- **THEN** the returned `ImageEntry` has `index == 3`

### Requirement: Annotation persistence
`ImageManager` SHALL save a list of `Polygon` objects for a given `ImageEntry` to `annotations/<basename>.json` and load them back correctly. After `save_annotation`, `is_annotated(entry)` SHALL return `True`. `load_annotation` SHALL return an empty list if no annotation file exists.

#### Scenario: Save and reload annotation
- **WHEN** `save_annotation(entry, polygons)` is called with a non-empty polygon list
- **THEN** a valid JSON file is written to `annotations/<basename>.json` and `is_annotated(entry)` returns `True`

#### Scenario: Load after restart
- **WHEN** `load_annotation(entry)` is called after application restart for an image with a saved annotation
- **THEN** it returns the same polygons that were saved, including `source`, `confidence`, and `points`

#### Scenario: Load non-existent annotation
- **WHEN** `load_annotation(entry)` is called for an image with no annotation file
- **THEN** it returns an empty list

#### Scenario: SAM polygon round-trip
- **WHEN** a polygon with `source = "sam_auto"` and `confidence = 0.87` is saved and loaded
- **THEN** the loaded polygon has identical `source` and `confidence` values

### Requirement: Annotation progress tracking
`ImageManager.get_progress()` SHALL return an `AnnotationProgress` with correct counts: `total`, `annotated`, `manual`, `auto_labeled`, and `unreviewed_auto`. `unreviewed_auto` SHALL be the count of images whose annotation file contains only SAM-sourced polygons AND have not been explicitly saved by the user in the current session.

#### Scenario: Mixed dataset progress
- **WHEN** a dataset has 10 images: 4 with manual annotations, 3 with SAM-only annotations (not re-saved by user), 1 with SAM annotations re-saved by user, 2 unannotated
- **THEN** `get_progress()` returns `total=10, annotated=8, manual=4, auto_labeled=4, unreviewed_auto=3`

#### Scenario: Empty dataset
- **WHEN** no images are loaded
- **THEN** `get_progress()` returns all zeros

### Requirement: Dataset split generation
`ImageManager.generate_split()` SHALL produce a `DatasetSplit` with train/val/test subsets following the ratios in `Config.dataset_split` (default 70/20/10), using a reproducible seed. The split SHALL be saved to `annotations/split.json`. If `split.json` already exists, `generate_split()` SHALL load and return the existing split without regenerating.

#### Scenario: First-time split generation
- **WHEN** `generate_split()` is called and no `split.json` exists
- **THEN** a `DatasetSplit` is returned with no image appearing in more than one split, and ratios approximate 70/20/10

#### Scenario: Idempotent split
- **WHEN** `generate_split()` is called twice on the same dataset
- **THEN** both calls return identical splits (same images in same subsets)

#### Scenario: No overlap between splits
- **WHEN** `generate_split()` is called on a dataset of 10 images
- **THEN** the union of `train`, `val`, and `test` equals all 10 images, with no duplicates
