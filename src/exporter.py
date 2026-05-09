from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import Config
    from src.image_manager import ImageEntry, ImageManager, Polygon


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def polygon_to_rle(points: list[tuple[float, float]], height: int, width: int) -> dict:
    import cv2
    from pycocotools import mask as coco_mask

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(binary_mask, [pts], 1)
    rle = coco_mask.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def compute_area_shoelace(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def compute_bbox(points: list[tuple[float, float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    def __init__(self, config: "Config") -> None:
        self._config = config

    # ------------------------------------------------------------------
    # COCO JSON
    # ------------------------------------------------------------------

    def export_single_coco(
        self,
        entry: "ImageEntry",
        polygons: "list[Polygon]",
        annotation_id_start: int = 1,
        use_rle: bool = True,
    ) -> dict:
        image_record = {
            "id": entry.index + 1,
            "file_name": entry.filename,
            "width": entry.width,
            "height": entry.height,
            "date_captured": "",
            "split": entry.split,
        }

        annotations = []
        ann_id = annotation_id_start
        for poly in polygons:
            ann = self._build_annotation(
                ann_id, entry.index + 1, poly, entry.height, entry.width, use_rle=use_rle
            )
            annotations.append(ann)
            ann_id += 1

        return {"images": [image_record], "annotations": annotations}

    def export_coco(
        self,
        image_manager: "ImageManager",
        output_path: str,
        split: str | None = None,
        use_rle: bool = True,
        progress_callback: callable = None,
    ) -> str:
        categories = self._build_categories()
        images_list: list[dict] = []
        annotations_list: list[dict] = []
        ann_id = 1

        entries = image_manager.get_all()
        total = len(entries)

        for i, entry in enumerate(entries):
            if split is not None and entry.split != split:
                continue

            polygons = image_manager.load_annotation(entry)
            if not polygons:
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            img_id = entry.index + 1
            images_list.append({
                "id": img_id,
                "file_name": entry.filename,
                "width": entry.width,
                "height": entry.height,
                "date_captured": "",
                "split": entry.split,
            })

            for poly in polygons:
                ann = self._build_annotation(
                    ann_id, img_id, poly, entry.height, entry.width, use_rle=use_rle
                )
                annotations_list.append(ann)
                ann_id += 1

            if progress_callback:
                progress_callback(i + 1, total)

        info_split = split if split else "all"
        output = {
            "info": {
                "description": "Daun Kelengkeh Itoh Dataset",
                "version": "1.0",
                "year": date.today().year,
                "contributor": "",
                "date_created": date.today().isoformat(),
                "split": info_split,
            },
            "licenses": [],
            "categories": categories,
            "images": images_list,
            "annotations": annotations_list,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return str(out_path)

    def export_coco_splits(
        self,
        image_manager: "ImageManager",
        output_dir: str,
        progress_callback: callable = None,
    ) -> dict[str, str]:
        out = Path(output_dir)
        result: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            file_path = out / f"{split_name}.json"
            result[split_name] = self.export_coco(
                image_manager,
                str(file_path),
                split=split_name,
                progress_callback=progress_callback,
            )
        return result

    # ------------------------------------------------------------------
    # Pascal VOC XML
    # ------------------------------------------------------------------

    def export_voc_xml(
        self,
        image_manager: "ImageManager",
        output_dir: str,
        split: str | None = None,
        progress_callback: callable = None,
    ) -> list[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        entries = image_manager.get_all()
        total = len(entries)

        for i, entry in enumerate(entries):
            if split is not None and entry.split != split:
                continue

            polygons = image_manager.load_annotation(entry)
            if not polygons:
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            xml_path = out / (Path(entry.filename).stem + ".xml")
            self._write_voc_xml(entry, polygons, xml_path)
            written.append(str(xml_path))

            if progress_callback:
                progress_callback(i + 1, total)

        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_categories(self) -> list[dict]:
        raw = getattr(self._config, "classes", [])
        result = []
        for cls in raw:
            if isinstance(cls, dict):
                result.append({
                    "id": cls["id"],
                    "name": cls["name"],
                    "supercategory": cls.get("supercategory", ""),
                })
        if not result:
            result = [{"id": 1, "name": "daun_kelengkeh_itoh", "supercategory": "plant"}]
        return result

    def _build_annotation(
        self,
        ann_id: int,
        image_id: int,
        poly: "Polygon",
        height: int,
        width: int,
        use_rle: bool = True,
    ) -> dict:
        points = poly.points
        bbox = compute_bbox(points)
        area = compute_area_shoelace(points)
        flat_poly = [coord for pt in points for coord in pt]

        if use_rle:
            segmentation = polygon_to_rle(points, height, width)
        else:
            segmentation = [flat_poly]

        occlusion_level_val = None
        if poly.occlusion_level is not None:
            occlusion_level_val = poly.occlusion_level.value

        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": poly.class_id,
            "segmentation": segmentation,
            "segmentation_polygon": [flat_poly],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "source": poly.source,
            "confidence": poly.confidence,
            "occlusion_level": occlusion_level_val,
            "occlusion_ratio": poly.occlusion_ratio,
        }

    def _write_voc_xml(
        self,
        entry: "ImageEntry",
        polygons: "list[Polygon]",
        xml_path: Path,
    ) -> None:
        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = entry.filename
        ET.SubElement(root, "path").text = entry.filepath

        source_el = ET.SubElement(root, "source")
        ET.SubElement(source_el, "database").text = "Daun Kelengkeh Itoh Dataset"

        size_el = ET.SubElement(root, "size")
        ET.SubElement(size_el, "width").text = str(entry.width)
        ET.SubElement(size_el, "height").text = str(entry.height)
        ET.SubElement(size_el, "depth").text = "3"

        ET.SubElement(root, "segmented").text = "1"

        for poly in polygons:
            bbox = compute_bbox(poly.points)
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = poly.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "source").text = poly.source
            ET.SubElement(obj, "confidence").text = str(poly.confidence)

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x_min)
            ET.SubElement(bndbox, "ymin").text = str(y_min)
            ET.SubElement(bndbox, "xmax").text = str(x_max)
            ET.SubElement(bndbox, "ymax").text = str(y_max)

            polygon_el = ET.SubElement(obj, "polygon")
            for idx, (x, y) in enumerate(poly.points, start=1):
                ET.SubElement(polygon_el, f"x{idx}").text = str(int(x))
                ET.SubElement(polygon_el, f"y{idx}").text = str(int(y))

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(xml_path), encoding="unicode", xml_declaration=False)
