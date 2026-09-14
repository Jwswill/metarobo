"""Hailo 장치에서 모델 로딩과 실제 추론만 검사한다 (모터/카메라 사용 없음).

실행: python3 scripts/debug_hailo_model.py
다른 모델: python3 scripts/debug_hailo_model.py --model-dir /path/to/best_hailo_model
"""

import argparse
from pathlib import Path
import sys
import time
import traceback


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "asset" / "best_hailo_model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    stage = "모델 파일 확인"
    try:
        model_dir = args.model_dir.expanduser().resolve()
        print(f"[1/4] {stage}: {model_dir}", flush=True)
        for name in ("best.hef", "metadata.yaml", "nms_config.json"):
            path = model_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"모델 파일이 없거나 비어 있음: {path}")
            print(f"  {name}: {path.stat().st_size:,} bytes", flush=True)

        stage = "HailoRT / Ultralytics import"
        print(f"[2/4] {stage}", flush=True)
        import hailo_platform
        import numpy as np
        import ultralytics
        from ultralytics import YOLO

        print(f"  Python: {sys.version.split()[0]}", flush=True)
        print(f"  Ultralytics: {ultralytics.__version__}", flush=True)
        print(f"  HailoRT: {getattr(hailo_platform, '__version__', '버전 정보 없음')}", flush=True)

        stage = "YOLO 모델 객체 생성"
        print(f"[3/4] {stage}", flush=True)
        model = YOLO(str(model_dir), task="detect")
        print("  객체 생성 완료. 실제 장치 동작은 다음 추론에서 확인합니다.", flush=True)

        stage = "Hailo 장치 초기화 및 실제 추론"
        print(f"[4/4] {stage}: 640x640 더미 이미지", flush=True)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        started = time.perf_counter()
        # 주행 노트북과 같은 설정. CPU는 호스트 전처리용이며 HEF 추론은 Hailo에서 수행.
        results = model.predict(frame, imgsz=640, conf=0.60, verbose=False, device="cpu")
        elapsed = time.perf_counter() - started
        if len(results) != 1 or results[0].boxes is None:
            raise RuntimeError("detect 결과 형식이 예상과 다릅니다.")
        names = results[0].names
        expected = {0: "safety", 1: "danger", 2: "corn"}
        actual = ({int(k): str(v).lower() for k, v in names.items()}
                  if isinstance(names, dict) else dict(enumerate(str(v).lower() for v in names)))
        if actual != expected:
            raise RuntimeError(f"클래스 불일치: expected={expected}, actual={actual}")
        print(f"  클래스: {actual}", flush=True)
        print(f"  검출 수: {len(results[0].boxes)} (더미 이미지에서는 0개여도 정상)", flush=True)
        print(f"  초기화 포함 추론 시간: {elapsed:.3f}s", flush=True)
        print("PASS: 모델 로딩, Hailo 장치 초기화, 실제 추론, 클래스 확인 완료.", flush=True)
        print("더미 이미지 검사이므로 실제 물체 인식 정확도는 검증하지 않습니다.", flush=True)
        return 0
    except Exception as exc:
        print(f"\nFAIL [{stage}]: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
