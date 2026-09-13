# 주행 영상 YOLO 라벨링 툴

주행 영상에서 프레임을 추출하고, 웹 브라우저에서 바운딩박스를 그려 YOLO 학습용 라벨(json)을 만드는 도구입니다.

- 클래스: `safety`(0), `danger`(1), `corn`(2)
- 결과 좌표: YOLO 정규화 포맷 (`x_center, y_center, width, height`, 0~1 범위)

## 0. 준비

```bash
pip install opencv-python
```

Python 3.9+ 권장. 라벨링 서버는 표준 라이브러리만 사용하므로 별도 설치가 필요 없습니다.

이 저장소에는 원본 영상(`driveDataset/`)과 추출된 프레임/라벨(`cap_image/`)이 포함되어 있지 않습니다 (`.gitignore` 처리, 용량 문제). 작업 전에 아래 구조로 직접 준비해주세요.

```
metarobo/
├── driveDataset/          # 라벨링할 원본 mp4 영상들을 여기에 넣기
├── cap_image/             # (자동 생성됨) 추출된 프레임 + 라벨
├── scripts/
│   └── extract_frames.py  # 영상 -> 프레임 추출
└── labeling_tool/
    ├── server.py           # 라벨링 서버
    └── static/             # 웹 UI
```

## 1. 프레임 추출

`driveDataset/` 폴더에 영상(mp4 등)을 넣고 실행:

```bash
python scripts/extract_frames.py
```

- 영상별로 **초당 5프레임**을 캡처해서 `cap_image/<영상파일명>/image/frame_000000.jpg ...` 로 저장합니다.
- 이미 추출된 영상은 자동으로 건너뜁니다. 다시 추출하려면 `--force` 옵션 사용.
- 옵션: `--input`(영상 폴더, 기본 `driveDataset`), `--output`(출력 폴더, 기본 `cap_image`), `--fps`(초당 프레임 수, 기본 5)

```bash
python scripts/extract_frames.py --fps 5 --force
```

실행하면 `cap_image/classes.txt`에 클래스 순서(`safety, danger, corn` = id 0,1,2)가 함께 기록됩니다.

## 2. 라벨링 툴 실행

```bash
python labeling_tool/server.py
```

브라우저에서 `http://localhost:8765` 접속 (포트를 바꾸려면 `PORT=9000 python labeling_tool/server.py`).

### 사용법

1. 상단 드롭다운에서 라벨링할 영상(디렉터리) 선택 — `(라벨링됨/전체)` 진행률이 함께 표시됩니다.
2. 왼쪽 **클래스 버튼**으로 클래스 선택 후, 캔버스에서 드래그해서 박스를 그립니다.
   - `1` = safety, `2` = danger, `3` = corn
3. 하단 **박스 목록**에서 항목 클릭 시 선택되고, 선택된 상태에서 클래스 버튼을 누르면 클래스가 바뀝니다.
   - `Delete`/`Backspace`: 선택된 박스 삭제
4. 이미지 이동: `A`/`←` 이전, `D`/`→` 다음 — 이동 시 변경 사항이 있으면 자동 저장됩니다.
5. `S`: 즉시 저장 (박스가 없는 프레임도 "객체 없음"으로 저장 가능)

왼쪽 이미지 목록에서 `✓숫자`는 저장된 박스 개수를 의미합니다.

## 3. 결과물 구조

```
cap_image/
├── classes.txt
└── <영상파일명>/
    ├── image/
    │   ├── frame_000000.jpg
    │   └── ...
    └── json/
        ├── frame_000000.json
        └── ...
```

`json` 파일 예시:

```json
{
  "image": "frame_000000.jpg",
  "width": 320,
  "height": 180,
  "annotations": [
    {
      "class_id": 0,
      "class_name": "safety",
      "bbox": { "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3 }
    }
  ]
}
```

`bbox`는 YOLO 포맷과 동일한 정규화 좌표(중심점, 너비, 높이 / 0~1)라서, YOLO 학습용 `.txt`로 변환할 때는 `class_id x_center y_center width height` 한 줄만 이어 쓰면 됩니다.
