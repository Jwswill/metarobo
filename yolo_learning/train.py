"""safety/danger/corn YOLOv8n 학습.

사전 준비:
  1. labeling_tool 로 cap_image 프레임 일부 이상 라벨링
  2. python yolo_learning/prepare_dataset.py 로 dataset/ 생성 (data.yaml 포함)

실행:
  python yolo_learning/train.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "dataset" / "data.yaml"
PRETRAINED = HERE / "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="safety_danger_corn")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"{DATA_YAML} 가 없습니다. 먼저 python yolo_learning/prepare_dataset.py 를 실행하세요."
        )

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(HERE / "runs"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
