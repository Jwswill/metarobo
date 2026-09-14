"""cap_image/의 라벨링 결과(json)를 YOLO 학습용 데이터셋으로 변환.

cap_image/<video>/image/*.jpg + cap_image/<video>/json/*.json (labeling_tool 결과)
  -> yolo_learning/dataset/images/{train,val}/*.jpg
     yolo_learning/dataset/labels/{train,val}/*.txt
     yolo_learning/dataset/data.yaml

json의 bbox(x_center,y_center,width,height, 0~1)가 이미 YOLO 정규화 포맷과 동일하므로
"class_id x_center y_center width height" 한 줄로 그대로 옮겨 적는다.

train/val은 프레임 단위가 아니라 촬영 구간(clip) 단위로 나눈다.
같은 구간에서 뽑은 연속 프레임은 서로 거의 똑같아서, 프레임 단위로 섞으면
학습/검증 데이터가 사실상 겹치는 데이터 유출(leakage)이 생기기 때문이다.

영상 폴더(cap_image/<video>) 하나 안에도 촬영 타임스탬프가 다른 여러 구간
(예: 20260914_004845_001.json, 20260914_004858_001.json ...)이 섞여 있을 수 있어서,
분할 단위는 "영상 폴더"가 아니라 "영상 폴더 + 촬영 타임스탬프 구간"으로 잡는다.
영상 폴더가 1개뿐이어도 그 안의 여러 구간으로 train/val을 나눌 수 있게 하기 위함이다.
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_IMAGE_DIR = ROOT / "cap_image"
OUT_DIR = Path(__file__).resolve().parent / "dataset"
CLASSES = ["safety", "danger", "corn"]  # cap_image/classes.txt, class_id 순서와 동일
VAL_RATIO = 0.2
SEED = 42

CLIP_PREFIX_RE = re.compile(r"^(\d{8}_\d{6})_")
MIN_SPLIT_CHUNK = 15  # 이보다 작은 조각으로는 더 쪼개지 않는다 (과도한 분절 방지)


def find_labeled_clips(video_dir: Path):
    """{clip_key: [(image_path, label_lines), ...]} 반환.

    clip_key는 "영상 폴더명::촬영 타임스탬프"로, 같은 영상 폴더 안이라도
    타임스탬프가 다른 촬영 구간은 서로 다른 clip으로 취급한다.
    파일명이 타임스탬프 패턴과 맞지 않으면 영상 폴더 전체를 하나의 clip으로 묶는다.
    """
    image_dir = video_dir / "image"
    json_dir = video_dir / "json"
    if not json_dir.is_dir():
        return {}

    clips = {}
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
        match = CLIP_PREFIX_RE.match(json_path.stem)
        clip_suffix = match.group(1) if match else video_dir.name
        clip_key = f"{video_dir.name}::{clip_suffix}"
        clips.setdefault(clip_key, []).append((image_path, lines))
    return clips


def split_oversized_clips(per_clip, max_clip_size):
    """max_clip_size보다 큰 clip은 촬영 순서를 유지한 채 앞/뒤 절반으로 나눈다.

    clip 하나가 지나치게 크면(예: 전체의 30%대) 그 clip 전체가 통째로
    train 또는 val 한쪽에만 들어가면서 목표 비율(val_ratio)을 크게 벗어나기 쉽다.
    그런 clip만 골라 절반으로 잘라 더 작은 단위로 만들면, 같은 촬영 구간이
    통째로 다른 split에 들어가는 leakage 위험을 최소화하면서도 비율을 맞출 수 있다.
    """
    result = {}
    stack = list(per_clip.items())
    while stack:
        key, frames = stack.pop()
        if len(frames) > max_clip_size and len(frames) >= MIN_SPLIT_CHUNK * 2:
            mid = len(frames) // 2
            stack.append((f"{key}#a", frames[:mid]))
            stack.append((f"{key}#b", frames[mid:]))
        else:
            result[key] = frames
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    video_dirs = [d for d in sorted(CAP_IMAGE_DIR.iterdir()) if d.is_dir()]
    per_clip = {}
    for d in video_dirs:
        per_clip.update(find_labeled_clips(d))
    clip_names = list(per_clip.keys())

    total_frames = sum(len(frames) for frames in per_clip.values())
    if not clip_names:
        print(
            f"[경고] cap_image 아래 라벨링된(json + annotations 존재) 프레임이 0개입니다.\n"
            f"       labeling_tool/server.py 로 최소한 일부 프레임에 박스를 그린 뒤 다시 실행하세요."
        )
        return

    # clip(촬영 구간) 단위로, 프레임 수 기준 val_ratio에 최대한 맞춰 배정한다.
    # (영상 폴더 개수가 아니라 프레임 수로 목표 비율을 맞춰야 특정 clip이 유난히
    #  크거나 작을 때도 train/val 비율이 한쪽으로 쏠리지 않는다.)
    max_clip_size = max(MIN_SPLIT_CHUNK, round(total_frames * min(args.val_ratio, 1 - args.val_ratio)))
    per_clip = split_oversized_clips(per_clip, max_clip_size)
    clip_names = list(per_clip.keys())

    rng = random.Random(args.seed)
    rng.shuffle(clip_names)
    clip_names.sort(key=lambda name: len(per_clip[name]), reverse=True)

    val_target = total_frames * args.val_ratio
    train_target = total_frames - val_target
    val_clips, train_clips = [], []
    val_count = train_count = 0
    for name in clip_names:
        n = len(per_clip[name])
        val_progress = val_count / val_target if val_target > 0 else float("inf")
        train_progress = train_count / train_target if train_target > 0 else float("inf")
        if val_progress <= train_progress:
            val_clips.append(name)
            val_count += n
        else:
            train_clips.append(name)
            train_count += n

    # clip이 2개 이상인데 한쪽이 비면(예: 매우 치우친 크기), 반대쪽에서 가장 작은
    # clip 하나를 옮겨 최소한 양쪽 다 데이터를 갖도록 한다.
    if len(clip_names) >= 2 and (not val_clips or not train_clips):
        donor, receiver = (train_clips, val_clips) if not val_clips else (val_clips, train_clips)
        donor.sort(key=lambda name: len(per_clip[name]))
        moved = donor.pop(0)
        receiver.append(moved)

    val_clips_set = set(val_clips)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for clip_name in clip_names:
        split = "val" if clip_name in val_clips_set else "train"
        safe_clip_name = clip_name.replace("::", "__")
        for image_path, lines in per_clip[clip_name]:
            stem = f"{safe_clip_name}__{image_path.stem}"
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

    print(f"전체 라벨링된 프레임: {total_frames}장 (영상 {len(video_dirs)}개, 구간(clip) {len(clip_names)}개)")
    print(f"train: {counts['train']}장 (구간 {sorted(train_clips)})")
    print(f"val:   {counts['val']}장 (구간 {sorted(val_clips)})")
    print(f"data.yaml 생성 완료: {data_yaml}")


if __name__ == "__main__":
    main()
