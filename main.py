import argparse
import logging
import sys
import warnings
from pathlib import Path

from src.config import Config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy third-party logs
    for noisy in ("PIL", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress harmless warnings from timm / mobile_sam internals
    warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
    warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")
    warnings.filterwarnings("ignore", message="Overwriting tiny_vit", category=UserWarning)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Labeling Daun Kelengkeh Itoh — Auto-Annotation Tool"
    )
    parser.add_argument("--images", type=str, help="Override folder gambar dari config.json")
    parser.add_argument("--sam-checkpoint", type=str, help="Override path SAM weights")
    parser.add_argument("--cpu", action="store_true", help="Paksa SAM berjalan di CPU")
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    cfg = Config.get()
    if args.images:
        cfg._ns.paths.images = Path(args.images)
    if args.sam_checkpoint:
        cfg._ns.paths.sam_checkpoint = Path(args.sam_checkpoint)
    if args.cpu:
        Config.force_cpu()


def _scaffold_dirs() -> None:
    cfg = Config.get()
    dirs = [
        cfg.paths.images,
        cfg.paths.annotations,
        cfg.paths.annotations_rendah,
        cfg.paths.annotations_sedang,
        cfg.paths.annotations_tinggi,
        cfg.paths.output_coco,
        cfg.paths.output_voc,
        cfg.paths.output_visualized,
        cfg.paths.sam_checkpoint.parent,  # sam_weights/
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    _setup_logging()
    log = logging.getLogger("main")
    args = _parse_args()

    log.info("Memulai aplikasi Labeling Daun Kelengkeh Itoh")
    Config.get()
    _apply_cli_overrides(args)
    _scaffold_dirs()
    log.info("Direktori output siap")

    from PyQt5.QtWidgets import QApplication
    from src.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(Config.get())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
