"""cap_image/의 라벨링 결과(json)를 YOLO 학습용 데이터셋으로 변환.

cap_image/<video>/image/*.jpg + cap_image/<video>/json/*.json (labeling_tool 결과)
  -> yolo_learning/dataset/images/{train,val}/*.jpg
     yolo_learning/dataset/labels/{train,val}/*.txt
     yolo_learning/dataset/data.yaml

json의 bbox(x_center,y_center,width,height, 0~1)가 이미 YOLO 정규화 포맷과 동일하므로
"class_id x_center y_center width height" 한 줄로 그대로 옮겨 적는다.

train/val은 프레임 단위가 아니라 영상(video) 단위로 나눈다.
같은 영상에서 뽑은 연속 프레임은 서로 거의 똑같아서, 프레임 단위로 섞으면
학습/검증 데이터가 사실상 겹치는 데이터 유출(leakage)이 생기기 때문이다.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_IMAGE_DIR = ROOT / "cap_image"
OUT_DIR = Path(__file__).resolve().parent / "dataset"
CLASSES = ["safety", "danger", "corn"]  # cap_image/classes.txt, class_id 순서와 동일
VAL_RATIO = 0.2
SEED = 42


def find_labeled_frames(video_dir: Path):
    """annotations가 1개 이상 있는 (image_path, label_lines) 목록 반환."""
    image_dir = video_dir / "image"
    json_dir = video_dir / "json"
    if not json_dir.is_dir():
        return []

    out = []
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        annotations = record.get("annotations", [])
        if not annotations:
            continue  # 객체 없음으로 저장된 프레임은 배경(negative) 샘플이라 학습에 넣어도 되지만,
            # 지금은 실제 박스가 있는 프레임만 우선 사용
        image_path = image_dir / record["image"]
        if not image_path.exists():
            continue
        lines = [
            f"{ann['class_id']} {ann['bbox']['x_center']} {ann['bbox']['y_center']} "
            f"{ann['bbox']['width']} {ann['bbox']['height']}"
            for ann in annotations
        ]
        out.append((image_path, lines))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    video_dirs = [d for d in sorted(CAP_IMAGE_DIR.iterdir()) if d.is_dir()]
    per_video = {d.name: find_labeled_frames(d) for d in video_dirs}
    labeled_videos = [name for name, frames in per_video.items() if frames]

    total_frames = sum(len(f) for f in per_video.values())
    if not labeled_videos:
        print(
            f"[경고] cap_image 아래 라벨링된(json + annotations 존재) 프레임이 0개입니다.\n"
            f"       labeling_tool/server.py 로 최소한 일부 프레임에 박스를 그린 뒤 다시 실행하세요."
        )
        return

    rng = random.Random(args.seed)
    rng.shuffle(labeled_videos)
    n_val_videos = max(1, round(len(labeled_videos) * args.val_ratio))
    val_videos = set(labeled_videos[:n_val_videos])
    train_videos = set(labeled_videos) - val_videos

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for video_name in labeled_videos:
        split = "val" if video_name in val_videos else "train"
        for image_path, lines in per_video[video_name]:
            stem = f"{video_name}__{image_path.stem}"
            shutil.copy2(image_path, OUT_DIR / "images" / split / f"{stem}{image_path.suffix}")
            with open(OUT_DIR / "labels" / split / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            counts[split] += 1

    data_yaml = OUT_DIR / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "path: {}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: {}\n"
            "names: {}\n".format(OUT_DIR.resolve(), len(CLASSES), CLASSES)
        )

    print(f"전체 라벨링된 프레임: {total_frames}장 (영상 {len(labeled_videos)}/{len(video_dirs)}개)")
    print(f"train: {counts['train']}장 (영상 {sorted(train_videos)})")
    print(f"val:   {counts['val']}장 (영상 {sorted(val_videos)})")
    print(f"data.yaml 생성 완료: {data_yaml}")


if __name__ == "__main__":
    main()
