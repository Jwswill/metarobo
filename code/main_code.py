from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import ipywidgets as widgets
from IPython.display import display
import time
import threading
from tiki.mini import TikiMini

# ===== 튜닝 파라미터 =====
LOWER_BLACK = np.array([0, 0, 0], dtype=np.uint8)
UPPER_BLACK = np.array([179, 255, 160], dtype=np.uint8)

LOWER_GREEN = np.array([42, 72, 90], dtype=np.uint8)
UPPER_GREEN = np.array([72, 199, 207], dtype=np.uint8)

BASE_SPEED = 140 # 직진 기본 속도 cx 위치랑 중앙이랑 차이가 DEAD_ZONE 값보다 작으면 직진임 ㅇㅇ
SCAN_BASE_SPEED = 100  # 1·7번째 후 ArUco SCAN 구간 직진/추종 속도
MAX_SPEED = 160 # 자세 조정할 때 모터 상한
KP = 0.30 # 자세 조정할 때 얼만큼 크게 꺾을건지 비율
KD = 0.12 # 짧은 시간에 빡세게 벗어날 때 얼마나 급하게 벗어나는지/복귀하는지 ratio
DEAD_ZONE = 18
MID_ERROR = 40 # 라인 많이 벗어나면 속도 줄이는 기준
ROI_OFFSET_X = -25 # ROI x축 위치 - 가운데가 기준임
ROI_WIDTH_RATIO = 0.40  # ROI 너비 전체 화면 너비 대비 비율임 ㅇㅇ
GREEN_OFFSET_Y = -20 # 초록 ROI y축 위치 (검정 ROI 중앙 기준)
MIN_CONTOUR_AREA = 80 # 감지된 검은색 중 라인 판별 시 최소 픽셀 수, 나머지는 노이즈 및 라인 아닌걸로 판별
MIN_GREEN_AREA = 60 # 바로 위와 같이 초록색 판별
LOOP_SLEEP = 0.03 # 프레임
DISPLAY_EVERY = 2 # 디스플레이 표시 프레임 주기

GREEN_PAUSE_SEC = 0.2 # 초록 보고 정지 시간
TURN_SPEED = 50 # 회전 속도
TURN_90_TICKS = 850 # 90도
TURN_180_TICKS = TURN_90_TICKS * 2  # 180도
TURN_270_TICKS = TURN_90_TICKS * 3  # "시계" 270도
TURN_TIMEOUT_SEC = 5.0
TURN_270_TIMEOUT_SEC = 8.0  # 270°는 더 김

# 0906 추가 - ArUco 마커 바로 판독 후 회전 시 다음 초록 판별 못하는 문제 대비 -> ARUCO_IMMEDIATE_TICKS 값 미만으로 이동 후 회전 시 잔류 초록 생각 안하고 초록 판별함
ARUCO_IMMEDIATE_TICKS = 200 # 

ARUCO_REQUIRED_COUNT = 4  # 한 프레임에 이 개수 이상일 때 합산·확정


def aruco_zone_from_count(count):
    """1·2번째 구간 → zone1 / 7·8번째 구간 → zone2."""
    if count in (1, 2):
        return 1
    if count in (7, 8):
        return 2
    return None


def commit_aruco_id_sum(marker_ids, zone):
    """4개(이상) ID 합산(중복 포함)만 ARUCO_SUMS에 저장. 로그는 G12에서."""
    if zone not in (1, 2):
        print(f"[dbg] ArUco 합산 스킵 (zone={zone}, count 구간 아님)")
        return False
    if len(marker_ids) < ARUCO_REQUIRED_COUNT:
        return False
    total = int(sum(marker_ids))
    ids_txt = ",".join(map(str, marker_ids))
    ARUCO_SUMS[zone] = total
    print(
        f"[dbg] ArUco {zone}구역 합 저장={total} "
        f"(ids={ids_txt}) | 현재 SUMS={dict(ARUCO_SUMS)}"
    )
    return True


def log_aruco_sums_once():
    """log_clear() 후 log(\"1구역합, 2구역합\") 한 번 (TIKI API)."""
    s1 = ARUCO_SUMS.get(1)
    s2 = ARUCO_SUMS.get(2)
    if s1 is None:
        print("[dbg] 경고: 1구역 ArUco 합 없음 → 0 사용")
        s1 = 0
    if s2 is None:
        print("[dbg] 경고: 2구역 ArUco 합 없음 → 0 사용")
        s2 = 0
    msg = f"{s1}, {s2}"
    tiki.log_clear()
    time.sleep(0.05)
    tiki.log(msg)
    print(f"[dbg] G12 log_clear + log → '{msg}' (1구역={s1}, 2구역={s2})")




prev_error = 0

# ===== 상태 =====
STATE_LINE = "LINE"              # 첫 초록 전: 라인 추종
STATE_SCAN = "SCAN"              # 1차/7차 초록 후: 라인 + ArUco
STATE_WAIT_ARUCO = "WAIT_ARUCO"  # 2차/7차 경로: 정지 + ArUco 대기
STATE_POST = "POST"              # 180° 이후: 라인 + 이후 초록
STATE_DONE = "DONE"              # 12번째 초록 후 정지

# ===== TIKI / 카메라 / ArUco =====
tiki = TikiMini()
tiki.set_motor_mode(tiki.MOTOR_MODE_PID)

# tiki.* 호출은 ESP32와 UART로 통신 → 메인 스레드(모터/엔코더)와 비전 스레드(부저)가
# 동시에 호출할 수 있는 구간(회전 중)에서는 이 락으로 감싸 직렬 통신이 섞이지 않게 한다.
tiki_lock = threading.Lock()

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (320, 180), "format": "BGR888"}
)
picam2.configure(camera_config)
picam2.start()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ===== 카메라 프레임 그래버 (스레드 공용) =====
# picam2.capture_array()를 여러 스레드가 동시에 직접 호출하면 카메라 드라이버 레벨에서
# 경합이 생길 수 있어, 캡처는 이 스레드 하나만 전담하고 나머지는 최신 프레임 복사본만 읽는다.
_frame_lock = threading.Lock()
_latest_frame = None
_stop_event = threading.Event()


def _camera_worker():
    global _latest_frame
    while not _stop_event.is_set():
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with _frame_lock:
            _latest_frame = frame


def get_frame():
    """최신 카메라 프레임의 복사본 반환 (아직 없으면 None)."""
    with _frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


camera_thread = threading.Thread(target=_camera_worker, daemon=True)
camera_thread.start()
while get_frame() is None:
    time.sleep(0.01)

# ===== YOLO 비전 (safety/danger/corn, cap_image/classes.txt 순서와 동일) =====
VISION_MODEL_PATH = "asset/best.pt"  # 학습 완료되면 이 경로에 가중치 파일 추가
VISION_CONF_THRESH = 0.5
CLASS_SAFETY, CLASS_DANGER, CLASS_CORN = 0, 1, 2
BUZZER_FREQ = 440  # corn 인식 시 울릴 주파수(Hz), 라 음
BUZZER_BEEP_SEC = 0.15  # corn 인식 시 짧게 울리는 시간

vision_model = YOLO(VISION_MODEL_PATH)


def beep_buzzer():
    """부저를 짧게 울림 (play_buzzer: PWM 주파수 제어, stop_buzzer: 정지).
    비전 스레드에서 호출되므로 tiki_lock으로 감싸 메인 스레드의 모터/엔코더 UART 통신과
    겹치지 않게 한다. sleep은 락 밖에서 해서 그 동안 메인 스레드가 블로킹되지 않게 한다."""
    with tiki_lock:
        tiki.play_buzzer(BUZZER_FREQ)
    time.sleep(BUZZER_BEEP_SEC)
    with tiki_lock:
        tiki.stop_buzzer()

video_widget = widgets.Image(
    format="jpeg",
    layout=widgets.Layout(width="320px", height="180px"),
)
status_widget = widgets.Label(value="준비 중...")
display(widgets.VBox([video_widget, status_widget]))


def convert_to_bytes(image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes() if ok else b""


def get_black_roi_box(height, width):
    y1, y2 = height // 2, height
    roi_w = max(int(width * ROI_WIDTH_RATIO), 1)
    x1 = (width - roi_w) // 2 + ROI_OFFSET_X
    x1 = int(np.clip(x1, 0, width - roi_w))
    x2 = x1 + roi_w
    return x1, y1, x2, y2


def get_green_roi_box(black_box):
    """검은선 ROI와 같은 x·너비, 높이=검은 ROI의 1/3, 세로 중앙 + GREEN_OFFSET_Y."""
    x1, y1, x2, y2 = black_box
    black_h = y2 - y1
    green_h = max(black_h // 3, 1)
    gy1 = y1 + (black_h - green_h) // 2 + GREEN_OFFSET_Y
    gy1 = int(np.clip(gy1, y1, y2 - green_h))
    gy2 = gy1 + green_h
    return x1, gy1, x2, gy2


def detect_black_line(frame):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = get_black_roi_box(height, width)
    black_box = (x1, y1, x2, y2)

    roi = frame[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cx = None
    center_x = roi_w // 2
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) >= MIN_CONTOUR_AREA:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)
                cv2.circle(roi, (cx, roi_h // 2), 4, (0, 0, 255), -1)

    cv2.line(roi, (center_x, 0), (center_x, roi_h), (255, 255, 0), 1)
    frame[y1:y2, x1:x2] = roi
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 165, 255), 2)
    return cx, center_x, black_box, frame


def detect_green(frame, black_box):
    """반환: green_found, frame, max_area, mask_pixels (HSV 마스크 픽셀 수)."""
    gx1, gy1, gx2, gy2 = get_green_roi_box(black_box)
    green_roi = frame[gy1:gy2, gx1:gx2].copy()

    hsv = cv2.cvtColor(green_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_pixels_raw = int(cv2.countNonZero(mask))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_pixels = int(cv2.countNonZero(mask))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    green_found = False
    max_area = 0.0
    if contours:
        max_g = max(contours, key=cv2.contourArea)
        max_area = float(cv2.contourArea(max_g))
        if max_area >= MIN_GREEN_AREA:
            green_found = True
            offset = np.array([[[gx1, gy1]]])
            cv2.drawContours(frame, [max_g + offset], -1, (0, 255, 0), 2)

    cv2.rectangle(frame, (gx1, gy1), (gx2 - 1, gy2 - 1), (0, 255, 0), 2)
    return green_found, frame, max_area, mask_pixels, mask_pixels_raw


def detect_aruco(frame):
    """ArUco ID 0~15 검출 (4_aruco_detection 참고). 반환: marker_ids, frame"""
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    marker_ids = []

    if ids is not None:
        valid_corners = []
        valid_ids = []
        for corner, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if 0 <= marker_id <= 15:
                marker_ids.append(marker_id)
                valid_corners.append(corner)
                valid_ids.append([marker_id])
        if valid_ids:
            cv2.aruco.drawDetectedMarkers(
                frame, valid_corners, np.array(valid_ids, dtype=np.int32)
            )

    return marker_ids, frame


def encoder_delta_avg(start_l, start_r):
    with tiki_lock:
        now_l = tiki.get_encoder(tiki.MOTOR_LEFT)
        now_r = tiki.get_encoder(tiki.MOTOR_RIGHT)
    return (abs(now_l - start_l) + abs(now_r - start_r)) / 2.0


def pause_then_turn_ccw(ticks, label="turn", timeout=None):
    """짧게 정지 후 get_encoder로 반시계 회전."""
    if timeout is None:
        timeout = TURN_TIMEOUT_SEC
    with tiki_lock:
        tiki.stop()
    time.sleep(GREEN_PAUSE_SEC)

    with tiki_lock:
        start_l = tiki.get_encoder(tiki.MOTOR_LEFT)
        start_r = tiki.get_encoder(tiki.MOTOR_RIGHT)
    t0 = time.time()

    with tiki_lock:
        tiki.counter_clockwise(TURN_SPEED)
    while encoder_delta_avg(start_l, start_r) < ticks:
        if time.time() - t0 > timeout:
            print(f"경고: 엔코더 {label} 타임아웃")
            break
        time.sleep(0.01)

    with tiki_lock:
        tiki.stop()


def pause_then_turn_cw(ticks, label="cw", timeout=None):
    """짧게 정지 후 get_encoder로 시계 회전.
    (회전 중 물체 인식은 vision_active로 게이팅된 별도 비전 스레드가 병행 수행)"""
    if timeout is None:
        timeout = TURN_TIMEOUT_SEC
    with tiki_lock:
        tiki.stop()
    time.sleep(GREEN_PAUSE_SEC)

    with tiki_lock:
        start_l = tiki.get_encoder(tiki.MOTOR_LEFT)
        start_r = tiki.get_encoder(tiki.MOTOR_RIGHT)
    t0 = time.time()

    with tiki_lock:
        tiki.clockwise(TURN_SPEED)
    while encoder_delta_avg(start_l, start_r) < ticks:
        if time.time() - t0 > timeout:
            print(f"경고: 엔코더 {label} 타임아웃")
            break
        time.sleep(0.01)

    with tiki_lock:
        tiki.stop()


def pause_then_turn_ccw_90():
    pause_then_turn_ccw(TURN_90_TICKS, "90")


def pause_then_turn_ccw_180():
    pause_then_turn_ccw(TURN_180_TICKS, "180")


def pause_then_turn_ccw_270():
    """반시계 270° — 회전 중 메인 초록 감지는 돌지 않음."""
    pause_then_turn_ccw(
        TURN_270_TICKS,
        label="CCW270",
        timeout=TURN_270_TIMEOUT_SEC,
    )


# ===== 비전 스레드 (회전 중에만 safety/danger/corn 추론) =====
# on_spin 콜백으로 회전 루프 안에서 동기 추론하면 추론이 걸리는 동안 엔코더 체크가
# 늦어져 회전각이 오버슈트될 수 있었다. 이제는 별도 스레드가 vision_active일 때만
# 최신 프레임을 계속 추론하고, 회전 루프는 엔코더만 보고 돌아 서로 블로킹하지 않는다.
vision_lock = threading.Lock()
vision_active = threading.Event()
turn_safety_seen = False
turn_danger_seen = False
turn_corn_seen = False


def _vision_worker():
    global turn_safety_seen, turn_danger_seen, turn_corn_seen
    prev_corn_seen = False
    while not _stop_event.is_set():
        if not vision_active.is_set():
            prev_corn_seen = False
            time.sleep(0.05)
            continue

        frame = get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        results = vision_model.predict(frame, conf=VISION_CONF_THRESH, verbose=False)
        class_ids = results[0].boxes.cls.numpy().astype(int).tolist()

        corn_now = CLASS_CORN in class_ids
        with vision_lock:
            if CLASS_SAFETY in class_ids:
                turn_safety_seen = True
            if CLASS_DANGER in class_ids:
                turn_danger_seen = True
            if corn_now:
                turn_corn_seen = True

        if corn_now and not prev_corn_seen:
            print("[dbg-vision] corn 인식 → 부저")
            beep_buzzer()
        prev_corn_seen = corn_now


vision_thread = threading.Thread(target=_vision_worker, daemon=True)
vision_thread.start()


def cw270_turn_with_vision(label):
    """CW 270° 회전 + 회전 중 safety/danger/corn YOLO 판별(비전 스레드가 병행 수행).
    회전이 끝나면 danger 우선으로 safety→'O' / danger→'X'를 SAFETY_LOG에 기록."""
    global turn_safety_seen, turn_danger_seen, turn_corn_seen
    with vision_lock:
        turn_safety_seen = False
        turn_danger_seen = False
        turn_corn_seen = False

    vision_active.set()
    pause_then_turn_cw(TURN_270_TICKS, label="270", timeout=TURN_270_TIMEOUT_SEC)
    vision_active.clear()

    with vision_lock:
        safety_seen, danger_seen, corn_seen = (
            turn_safety_seen,
            turn_danger_seen,
            turn_corn_seen,
        )

    if danger_seen:
        result = "X"
    elif safety_seen:
        result = "O"
    else:
        result = "?"
    SAFETY_LOG.append(result)
    print(
        f"[dbg-vision] {label} 회전 판정 → {result} "
        f"(safety={safety_seen}, danger={danger_seen}, corn={corn_seen}) "
        f"| SAFETY_LOG={SAFETY_LOG}"
    )


def arm_after_turn(frame, black_box, label="turn"):
    """회전 직후: 잔류 초록이 있으면 prev=True(이탈 대기), 없으면 prev=False(다음 상승엣지 바로 가능)."""
    global green_armed, prev_green
    green_found, frame, _area, _pix, _pix_raw = detect_green(frame, black_box)
    green_armed = True
    prev_green = bool(green_found)
    print(
        f"[dbg] {label} 후 arm | residual={green_found} "
        f"armed={green_armed} prev={prev_green}"
    )
    return green_found, frame


def finish_ccw90_ignore_next(frame, black_box, ignore_count):
    """CCW90 직후: 바로 보이는 초록=ignore_count번째(+1만 하고 통과). 없으면 상승엣지 대기."""
    global green_armed, prev_green, g3_saw_clear, g3_block_logged
    green_now, frame, area, _pix, _raw = detect_green(frame, black_box)
    g3_saw_clear = False
    g3_block_logged = False
    if green_now:
        print(
            f"[dbg] CCW90 직후 초록 area={area:.0f} → {ignore_count}번째 즉시 +1 (무시)"
        )
        set_green_count(
            ignore_count, f"{ignore_count}번째 초록 (CCW90 직후 즉시) → 무시"
        )
        green_armed = False
        prev_green = True
        print(f"[dbg] count={ignore_count}, 이탈 후 다음 엣지 대기")
    else:
        green_armed = True
        prev_green = False
        print(f"[dbg] CCW90 직후 초록 없음 → 상승엣지로 {ignore_count}번째 대기")
    return frame


def finish_g3_turn(frame, black_box):
    """G3 CCW90 직후 → 4번째 무시 처리."""
    return finish_ccw90_ignore_next(frame, black_box, 4)


def drive_by_error(cx, center_x, base_speed=None):
    """base_speed: None이면 BASE_SPEED. SCAN(ArUco) 구간만 SCAN_BASE_SPEED 전달."""
    global prev_error
    spd = BASE_SPEED if base_speed is None else base_speed

    if cx is None:
        prev_error = 0
        tiki.stop()
        return "LOST", 0, spd

    error = cx - center_x
    abs_err = abs(error)

    if abs_err <= DEAD_ZONE:
        prev_error = error
        tiki.forward(spd)
        return "FORWARD", error, spd

    if abs_err > MID_ERROR:
        base = int(spd * 0.70)
        kp = KP * 1.35
    else:
        base = int(spd * 0.90)
        kp = KP

    derivative = error - prev_error
    turn = int(kp * error + KD * derivative)
    prev_error = error

    left_speed = int(np.clip(base + turn, 0, MAX_SPEED))
    right_speed = int(np.clip(base - turn, 0, MAX_SPEED))
    tiki.set_motor_power(tiki.MOTOR_LEFT, left_speed)
    tiki.set_motor_power(tiki.MOTOR_RIGHT, right_speed)

    action = "LEFT" if error < 0 else "RIGHT"
    return action, error, base



    if old != GREEN_OFFSET_Y:
        print(f"[dbg] GREEN_OFFSET_Y {old} → {GREEN_OFFSET_Y} (after count={count})")


def set_green_count(new_count, reason=""):
    """초록 카운트 변경 시 항상 print."""
    global green_count
    old = green_count
    green_count = new_count
    if old != green_count:
        msg = f"[green_count] {old} → {green_count}"
        if reason:
            msg += f" | {reason}"
        print(msg)
    return green_count


green_count = 0
# zone(1/2) → 해당 구간 ArUco ID 합 (mutable dict, 전역 섀도잉 방지)
ARUCO_SUMS = {1: None, 2: None}

# CW270 회전(G5/G6/G11)마다 safety→'O' / danger→'X' 판정 기록
SAFETY_LOG = []

try:
    frame_i = 0
    state = STATE_LINE
    prev_green = False
    green_armed = True  # 같은 초록을 연속 카운트하지 않도록
    scan_start_l = 0
    scan_start_r = 0
    # G3 이후 실제 4번째가 엣지로 안 잡히는지 확인용
    g3_saw_clear = False      # count==3에서 found=False를 한 번이라도 봤는지
    g3_block_logged = False   # 잔류에 막힌 초록 로그 중복 방지
    ARUCO_SUMS[1] = None
    ARUCO_SUMS[2] = None

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(LOOP_SLEEP)
            continue

        cx, center_x, black_box, frame = detect_black_line(frame)
        green_found, frame, green_area, green_pixels, green_pixels_raw = detect_green(frame, black_box)

        # 초록 상승엣지 + 디버그
        if not green_found:
            green_armed = True
        green_edge = green_found and not prev_green and green_armed

        # --- count==3 (G3 후~코드 4번째 전): 4번째 먹힘 여부 판별 ---
        if state == STATE_POST and green_count == 3:
            if not green_found:
                if not g3_saw_clear:
                    g3_saw_clear = True
                    g3_block_logged = False
                    print(
                        "[dbg-g3] 이탈 확인 found=False "
                        "→ 이제부터 다음 상승엣지 = 코드상 4번째 후보"
                    )
            elif green_found and not green_edge:
                # 초록은 보이는데 카운트 안 됨 (잔류 prev 또는 armed)
                why = []
                if prev_green:
                    why.append("prev=True(상승엣지 아님·잔류연속)")
                if not green_armed:
                    why.append("armed=False")
                if not g3_block_logged or frame_i % 15 == 0:
                    g3_block_logged = True
                    print(
                        f"[dbg-g3] 초록 있는데 edge=False (실제 4번째가 여기면 먹힘) "
                        f"why={'+'.join(why) if why else '?'} "
                        f"area={green_area:.0f} raw={green_pixels_raw} "
                        f"clear_seen={g3_saw_clear}"
                    )

        if green_found != prev_green:
            print(
                f"[dbg-green] found={green_found} prev={prev_green} "
                f"armed={green_armed} edge={green_edge} "
                f"count={green_count} state={state} "
                f"area={green_area:.0f} pix={green_pixels} raw={green_pixels_raw}"
            )
        elif green_found and not green_armed and frame_i % 20 == 0:
            print(
                f"[dbg-green] STUCK? found=True armed=False "
                f"count={green_count} state={state} "
                f"area={green_area:.0f} pix={green_pixels} raw={green_pixels_raw}"
            )
        elif (
            state == STATE_POST
            and not green_found
            and green_pixels_raw > 0
            and frame_i % 10 == 0
        ):
            print(
                f"[dbg-green] HSV는 있는데 found=False "
                f"raw={green_pixels_raw} pix={green_pixels} "
                f"area={green_area:.0f} min={MIN_GREEN_AREA} count={green_count}"
            )
        elif state == STATE_POST and frame_i % 20 == 0:
            print(
                f"[dbg-green] POST poll found={green_found} area={green_area:.0f} "
                f"pix={green_pixels} raw={green_pixels_raw} min={MIN_GREEN_AREA} "
                f"armed={green_armed} edge={green_edge} count={green_count} "
                f"g3_clear={g3_saw_clear}"
            )

        prev_green = green_found

        marker_ids = []
        action, error, base = state, 0, 0

        if state == STATE_LINE:
            if green_edge:
                set_green_count(1, "1번째 초록 → CCW 90°")
                status_widget.value = f"G{green_count}: CCW90"
                video_widget.value = convert_to_bytes(frame)
                pause_then_turn_ccw_90()
                state = STATE_SCAN
                frame = get_frame()
                cx, center_x, black_box, frame = detect_black_line(frame)
                _, frame = arm_after_turn(frame, black_box, "SCAN")
                scan_start_l = tiki.get_encoder(tiki.MOTOR_LEFT)
                scan_start_r = tiki.get_encoder(tiki.MOTOR_RIGHT)
                print(f"[dbg] → SCAN | count={green_count}")
                action = "CCW90 DONE"
            else:
                action, error, base = drive_by_error(cx, center_x)

        elif state == STATE_SCAN:
            marker_ids, frame = detect_aruco(frame)

            if len(marker_ids) >= ARUCO_REQUIRED_COUNT:
                marker_text = ", ".join(map(str, marker_ids))
                zone = aruco_zone_from_count(green_count)
                commit_aruco_id_sum(marker_ids, zone)
                traveled = encoder_delta_avg(scan_start_l, scan_start_r)
                immediate_aruco = traveled < ARUCO_IMMEDIATE_TICKS
                print(
                    f"ArUco 인식(라인 중): {marker_text} → CCW 180° "
                    f"| count={green_count} traveled={traveled:.0f} "
                    f"immediate={immediate_aruco}"
                )
                pause_then_turn_ccw_180()

                if green_count == 1:
                    # 1번째 후 SCAN: ArUco → count+1(=2) → POST, 필요 시 3 즉시
                    set_green_count(
                        green_count + 1,
                        f"ArUco 인식(라인 중) → CCW 180° | {marker_text}",
                    )
                    state = STATE_POST
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    green_now, frame, _a, _p, _pr = detect_green(frame, black_box)

                    if immediate_aruco and green_now:
                        print("[dbg] 제자리 ArUco 경로 → 180° 직후 초록을 3번째로 처리")
                        set_green_count(3, "3번째 초록 (180° 직후 즉시)")
                        green_armed = False
                        status_widget.value = f"G{green_count}: CCW90"
                        video_widget.value = convert_to_bytes(frame)
                        pause_then_turn_ccw_90()
                        frame = get_frame()
                        cx, center_x, black_box, frame = detect_black_line(frame)
                        frame = finish_g3_turn(frame, black_box)
                        action = "G3 CCW90 (immediate after 180)"
                    else:
                        green_armed = True
                        prev_green = bool(green_now)
                        print(
                            f"[dbg] → POST arm | residual={green_now} "
                            f"armed={green_armed} prev={prev_green}"
                        )
                        action = f"ARUCO {marker_text} + CCW180"
                    print(f"[dbg] → POST 진입 | count={green_count}")

                elif green_count == 7:
                    # 7번째 후 SCAN: ArUco → count+1(=8, 2와 동일) → POST, 필요 시 9 즉시(3과 동일)
                    set_green_count(
                        green_count + 1,
                        f"ArUco 인식(G7 SCAN) → CCW 180° | {marker_text}",
                    )
                    state = STATE_POST
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    green_now, frame, _a, _p, _pr = detect_green(frame, black_box)

                    if immediate_aruco and green_now:
                        print("[dbg] G7 ArUco 직후 초록 → 9번째(3과 동일) 즉시")
                        set_green_count(9, "9번째 초록 (180° 직후 즉시) → CCW 90°")
                        green_armed = False
                        status_widget.value = f"G{green_count}: CCW90"
                        video_widget.value = convert_to_bytes(frame)
                        pause_then_turn_ccw_90()
                        frame = get_frame()
                        cx, center_x, black_box, frame = detect_black_line(frame)
                        frame = finish_ccw90_ignore_next(frame, black_box, 10)
                        action = "G9 CCW90 (immediate after G7 ArUco)"
                    else:
                        green_armed = True
                        prev_green = bool(green_now)
                        print(
                            f"[dbg] → POST after G7 ArUco | residual={green_now} "
                            f"armed={green_armed} prev={prev_green} count={green_count}"
                        )
                        action = f"G7 ARUCO {marker_text} + CCW180"
                    print(f"[dbg] → POST 진입 | count={green_count}")
                else:
                    action = f"ARUCO {marker_text} (unexpected count={green_count})"

            elif green_edge and green_count in (1, 7):
                if green_count == 1:
                    set_green_count(2, "2번째 초록 → ArUco 대기")
                else:
                    set_green_count(8, "8번째 초록 → ArUco 대기 (2와 동일)")
                green_armed = False
                tiki.stop()
                state = STATE_WAIT_ARUCO
                action = "WAIT ARUCO"
            else:
                action, error, base = drive_by_error(
                    cx, center_x, SCAN_BASE_SPEED
                )

        elif state == STATE_WAIT_ARUCO:
            tiki.stop()
            marker_ids, frame = detect_aruco(frame)
            if len(marker_ids) >= ARUCO_REQUIRED_COUNT:
                marker_text = ", ".join(map(str, marker_ids))
                zone = aruco_zone_from_count(green_count)
                commit_aruco_id_sum(marker_ids, zone)
                print(
                    f"ArUco 인식(대기 중): {marker_text} → CCW 180° "
                    f"(count 유지={green_count})"
                )
                pause_then_turn_ccw_180()
                state = STATE_POST
                frame = get_frame()
                cx, center_x, black_box, frame = detect_black_line(frame)
                if green_count == 8:
                    # 2번 WAIT 경로와 동일 패턴 → 직후 초록이면 9(3과 동일)
                    green_now, frame, _a, _p, _pr = detect_green(frame, black_box)
                    if green_now:
                        print("[dbg] G8 WAIT ArUco 직후 초록 → 9번째 즉시")
                        set_green_count(9, "9번째 초록 (WAIT ArUco 직후) → CCW 90°")
                        green_armed = False
                        status_widget.value = f"G{green_count}: CCW90"
                        video_widget.value = convert_to_bytes(frame)
                        pause_then_turn_ccw_90()
                        frame = get_frame()
                        cx, center_x, black_box, frame = detect_black_line(frame)
                        frame = finish_ccw90_ignore_next(frame, black_box, 10)
                        action = "G9 CCW90 (after WAIT ArUco)"
                    else:
                        _, frame = arm_after_turn(frame, black_box, "POST")
                        action = f"ARUCO {marker_text} + CCW180"
                else:
                    _, frame = arm_after_turn(frame, black_box, "POST")
                    action = f"ARUCO {marker_text} + CCW180"
                print(f"[dbg] → POST 진입 | count={green_count}")
            else:
                action = "WAIT ARUCO"

        elif state == STATE_POST:
            # 180° 이후 라인 추종 + 3~10번째 초록
            if green_edge:
                print(
                    f"[dbg] POST green_edge! count {green_count}→{green_count+1} "
                    f"armed={green_armed}"
                )
                nxt = green_count + 1
                if nxt == 3:
                    reason = "3번째 초록 → CCW 90°"
                elif nxt == 4:
                    reason = "4번째 초록 → 무시"
                elif nxt == 5:
                    reason = "5번째 초록 → CW 270° (회전 중 초록 미감지)"
                elif nxt == 6:
                    reason = "6번째 초록 → CW 270° (회전 중 초록 미감지)"
                elif nxt == 7:
                    reason = "7번째 초록 → CCW 90° + ArUco 스캔 (1과 동일)"
                elif nxt == 8:
                    reason = "8번째 초록 → ArUco 관련 (2와 동일, POST 엣지면 주의)"
                elif nxt == 9:
                    reason = "9번째 초록 → CCW 90° (3과 동일)"
                elif nxt == 10:
                    reason = "10번째 초록 → 무시 (4와 동일)"
                elif nxt == 11:
                    reason = "11번째 초록 → CW 270° (회전 중 초록 미감지)"
                elif nxt == 12:
                    reason = "12번째 초록 → 정지"
                else:
                    reason = f"{nxt}번째 초록"
                if nxt == 4:
                    print(
                        f"[dbg-g3] ★ 코드 3→4 엣지 발생 | "
                        f"그전에 found=False 이탈 있었음? {g3_saw_clear}"
                    )
                set_green_count(nxt, reason)
                green_armed = False
                if green_count == 3:
                    status_widget.value = f"G{green_count}: CCW90"
                    video_widget.value = convert_to_bytes(frame)
                    pause_then_turn_ccw_90()
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    frame = finish_g3_turn(frame, black_box)
                    action = "G3 CCW90"
                elif green_count == 4:
                    action, error, base = drive_by_error(cx, center_x)
                elif green_count in (5, 6):
                    status_widget.value = f"G{green_count}: CW270"
                    video_widget.value = convert_to_bytes(frame)
                    cw270_turn_with_vision(f"G{green_count}")
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    # 회전 중/직후 잔류 초록은 카운트하지 않음
                    _, frame = arm_after_turn(
                        frame, black_box, f"G{green_count}"
                    )
                    action = f"G{green_count} CW270 DONE"
                elif green_count == 7:
                    status_widget.value = f"G{green_count}: CCW90→SCAN"
                    video_widget.value = convert_to_bytes(frame)
                    pause_then_turn_ccw_90()
                    state = STATE_SCAN
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    _, frame = arm_after_turn(frame, black_box, "G7-SCAN")
                    scan_start_l = tiki.get_encoder(tiki.MOTOR_LEFT)
                    scan_start_r = tiki.get_encoder(tiki.MOTOR_RIGHT)
                    print(f"[dbg] → SCAN (G7=1과 동일) | count={green_count}")
                    action = "G7 CCW90 → SCAN"
                elif green_count == 8:
                    # 2번은 보통 SCAN/ArUco에서 처리. POST에서 오면 라인만
                    action, error, base = drive_by_error(cx, center_x)
                elif green_count == 9:
                    status_widget.value = f"G{green_count}: CCW90"
                    video_widget.value = convert_to_bytes(frame)
                    pause_then_turn_ccw_90()
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    frame = finish_ccw90_ignore_next(frame, black_box, 10)
                    action = "G9 CCW90"
                elif green_count == 10:
                    action, error, base = drive_by_error(cx, center_x)
                elif green_count == 11:
                    status_widget.value = f"G{green_count}: CW270"
                    video_widget.value = convert_to_bytes(frame)
                    cw270_turn_with_vision(f"G{green_count}")
                    frame = get_frame()
                    cx, center_x, black_box, frame = detect_black_line(frame)
                    _, frame = arm_after_turn(frame, black_box, "G11")
                    action = "G11 CW270 DONE"
                elif green_count == 12:
                    tiki.stop()
                    log_aruco_sums_once()
                    state = STATE_DONE
                    status_widget.value = "G12: STOP DONE"
                    print("[dbg] 12번째 초록 → 로그 출력 후 정지 (DONE)")
                    action = "G12 STOP+LOG"
                else:
                    action, error, base = drive_by_error(cx, center_x)
            else:
                action, error, base = drive_by_error(cx, center_x)

        elif state == STATE_DONE:
            tiki.stop()
            action = "DONE"

        if frame_i % DISPLAY_EVERY == 0:
            aruco_txt = (
                ",".join(map(str, marker_ids)) if marker_ids else "-"
            )
            status_widget.value = (
                f"{state} | G={green_count} | {action} | "
                f"cx={cx} err={error} | ArUco={aruco_txt}"
            )
            video_widget.value = convert_to_bytes(frame)

        frame_i += 1
        time.sleep(LOOP_SLEEP)

except KeyboardInterrupt: 
    status_widget.value = "중단됨"
    print("라인트레이싱 중단")

finally:
    with tiki_lock:
        tiki.stop()
    _stop_event.set()
    camera_thread.join(timeout=2.0)
    vision_thread.join(timeout=2.0)
    picam2.stop()
    picam2.close()
    print("모터 정지 / 카메라 종료 / 스레드 정리 완료")