"""driveDataset의 영상들을 초당 지정 FPS로 캡처해서 cap_image/<영상이름>/image/ 에 저장한다.

사용법:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --fps 5 --force
"""
import argparse
import glob
import os

import cv2

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def extract(video_path: str, out_root: str, target_fps: float, force: bool) -> None:
    name = os.path.splitext(os.path.basename(video_path))[0]
    image_dir = os.path.join(out_root, name, "image")
    json_dir = os.path.join(out_root, name, "json")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if not force and os.listdir(image_dir):
        print(f"[skip] {name} (이미 추출됨, --force로 재추출)")
        return

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if src_fps <= 0:
        src_fps = 30.0
        print(f"[warn] {name}: fps 정보를 읽을 수 없어 30으로 가정")

    interval = max(1, round(src_fps / target_fps))

    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(image_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()

    print(f"[done] {name}: {frame_idx} frames -> {saved_idx} saved (src_fps={src_fps:.2f}, interval={interval})")


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 -> 프레임 캡처 (라벨링용)")
    parser.add_argument("--input", default="driveDataset", help="영상 디렉터리")
    parser.add_argument("--output", default="cap_image", help="프레임 출력 디렉터리")
    parser.add_argument("--fps", type=float, default=5.0, help="초당 캡처할 프레임 수")
    parser.add_argument("--force", action="store_true", help="이미 추출된 영상도 다시 추출")
    args = parser.parse_args()

    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
    videos.sort()

    if not videos:
        print(f"'{args.input}'에서 영상을 찾지 못했습니다.")
        return

    os.makedirs(args.output, exist_ok=True)

    # YOLO 클래스 순서를 고정해서 기록해 둔다 (라벨링 툴과 동일하게 유지)
    classes_path = os.path.join(args.output, "classes.txt")
    if not os.path.exists(classes_path):
        with open(classes_path, "w", encoding="utf-8") as f:
            f.write("safety\ndanger\ncorn\n")

    for video_path in videos:
        extract(video_path, args.output, args.fps, args.force)


if __name__ == "__main__":
    main()
