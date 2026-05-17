import os

from roboflow import Roboflow
from tqdm import tqdm

from a2.consts import SUPPORTED_IMAGE_FORMATS, ROBOFLOW_API_KEY
from a2.utils import flatten_lists, get_directory_content, dump_to_json, get_cli_arguments


def annotate(
        image_paths: list[str],
        target_annotation_directory: str,
        roboflow_api_key: str,
        roboflow_project_id: str,
        roboflow_project_version: int,
        detection_confidence_threshold: int = 40,
        detection_iou_threshold: int = 30
) -> None:
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace().project(roboflow_project_id)
    model = project.version(roboflow_project_version).model

    for image_source_path in tqdm(image_paths):
        annotations = model.predict(
            image_path=image_source_path,
            confidence=detection_confidence_threshold
        ).json()
        source_image_file_name = os.path.basename(image_source_path)
        file_name = os.path.splitext(source_image_file_name)[0]
        target_json_file_name = f"{file_name}.json"
        target_json_path = os.path.join(target_annotation_directory, target_json_file_name)
        dump_to_json(target_json_path, annotations)


def annotate_legacy(
        source_image_directory: str,
        target_annotation_directory: str,
        roboflow_api_key: str,
        roboflow_project_id: str,
        roboflow_project_version: int,
        detection_confidence_threshold: int = 40,
        detection_iou_threshold: int = 30
) -> None:
    image_source_paths = flatten_lists([
        get_directory_content(directory_path=source_image_directory, extension=extension)
        for extension
        in SUPPORTED_IMAGE_FORMATS
    ])
    annotate(
        image_paths=image_source_paths,
        target_annotation_directory=target_annotation_directory,
        roboflow_api_key=roboflow_api_key,
        roboflow_project_id=roboflow_project_id,
        roboflow_project_version=roboflow_project_version,
        detection_confidence_threshold=detection_confidence_threshold,
        detection_iou_threshold=detection_iou_threshold
    )


if __name__ == "__main__":
    args = get_cli_arguments()

    roboflow_api_key = args.roboflow_api_key if args.roboflow_api_key else os.environ.get(ROBOFLOW_API_KEY)
    if roboflow_api_key is None:
        raise Exception("Please set ROBOFLOW_API_KEY environment variable.")

    # Determine image paths based on input mode
    image_paths = []

    if args.source_image_directory:
        # Legacy mode: scan directory for images
        image_paths = flatten_lists([
            get_directory_content(directory_path=args.source_image_directory, extension=extension)
            for extension in SUPPORTED_IMAGE_FORMATS
        ])
    elif args.image_path:
        # Single image mode
        image_paths = [args.image_path]
    elif args.image_list:
        # Batch mode: read from text file
        with open(args.image_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    image_paths.append(line)

    if not image_paths:
        raise Exception("No images found to annotate.")

    annotate(
        image_paths=image_paths,
        target_annotation_directory=args.target_annotation_directory,
        roboflow_api_key=roboflow_api_key,
        roboflow_project_id=args.roboflow_project_id,
        roboflow_project_version=args.roboflow_project_version,
        detection_confidence_threshold=args.detection_confidence_threshold,
        detection_iou_threshold=args.detection_iou_threshold
    )
