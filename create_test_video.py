import cv2

input_file = "video/badminton_test.mp4"
output_file = "video/badminton_test_10sec.mp4"

cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_to_copy = int(fps * 10)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (width, height)
)

count = 0

while count < frames_to_copy:
    ret, frame = cap.read()

    if not ret:
        break

    out.write(frame)
    count += 1

cap.release()
out.release()

print("=" * 60)
print("10-SECOND TEST VIDEO CREATED")
print("=" * 60)
print(f"FPS       : {fps}")
print(f"Resolution: {width} x {height}")
print(f"Frames    : {count}")
print(f"Output    : {output_file}")
print("=" * 60)
