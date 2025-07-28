import cv2
import time

# 카메라 열기
cap = cv2.VideoCapture(0)

# 프레임을 하나 읽어서 크기를 확인
ret, frame = cap.read()
if not ret:
    print("카메라에서 프레임을 읽을 수 없습니다.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# 비디오 저장 설정 (mp4v 코덱 + 프레임 크기 일치)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# 시작 시간 저장
start_time = time.time()

# 첫 프레임 저장
out.write(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 더 이상 읽을 수 없습니다.")
        break

    out.write(frame)

    if time.time() - start_time >= 10:
        print("10초 경과. 스트리밍을 종료합니다.")
        break

cap.release()
out.release()

