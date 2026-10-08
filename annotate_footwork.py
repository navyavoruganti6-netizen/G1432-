import cv2
import os
import glob

FRAME_FOLDER = "footwork_frames"
LABEL_FOLDER = "footwork_labels"

CLASSES = [
    "forehand_front_corner",
    "backhand_front_corner",
    "forehand_side",
    "backhand_side",
    "forehand_back_court_corner",
    "backhand_back_court_corner"
]

os.makedirs(LABEL_FOLDER, exist_ok=True)

frames = sorted(glob.glob(os.path.join(FRAME_FOLDER, "*.jpg")))

print("Frames found:", len(frames))
print("Annotation program ready.")

current_frame = 0
selected_class = None
drawing = False
start_point = None
end_point = None
image = None
display_image = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        cv2.rectangle(
            display_image,
            start_point,
            end_point,
            (0, 255, 0),
            2
        )

def save_annotation():
    global start_point, end_point, selected_class

    if selected_class is None:
        print("Please select a class first.")
        return

    if start_point is None or end_point is None:
        print("Please draw a box around the player.")
        return

    h, w = image.shape[:2]

    x1, y1 = start_point
    x2, y2 = end_point

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    scale_x = w / display_image.shape[1]
    scale_y = h / display_image.shape[0]

    x1 = int(x1 * scale_x)
    x2 = int(x2 * scale_x)
    y1 = int(y1 * scale_y)
    y2 = int(y2 * scale_y)

    box_w = x2 - x1
    box_h = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

def show_frame():
    global image, display_image, start_point, end_point

    start_point = None
    end_point = None

    image = cv2.imread(frames[current_frame])

    if image is None:
        print("Could not open:", frames[current_frame])
        return

    display_image = image.copy()

    cv2.namedWindow("Footwork Annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Footwork Annotation", 1200, 700)

    cv2.setMouseCallback(
        "Footwork Annotation",
        mouse_callback
    )

    print("\nFrame:", os.path.basename(frames[current_frame]))
    print("Press 1-6 to select class")
    print("Press S to save")
    print("Press N for next frame")
    print("Press P for previous frame")
    print("Press Q to quit")

    while True:
        cv2.imshow("Footwork Annotation", display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            selected_class = CLASSES[0]
            print("Selected:", selected_class)

        elif key == ord("2"):
            selected_class = CLASSES[1]
            print("Selected:", selected_class)

        elif key == ord("3"):
            selected_class = CLASSES[2]
            print("Selected:", selected_class)

        elif key == ord("4"):
            selected_class = CLASSES[3]
            print("Selected:", selected_class)

        elif key == ord("5"):
            selected_class = CLASSES[4]
            print("Selected:", selected_class)

        elif key == ord("6"):
            selected_class = CLASSES[5]
            print("Selected:", selected_class)

        elif key == ord("s"):
            save_annotation()

        elif key == ord("n"):
            break

        elif key == ord("p"):
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            return

while current_frame < len(frames):

    show_frame()

    key = cv2.waitKey(0) & 0xFF

    if key == ord("n"):
        current_frame += 1

    elif key == ord("p"):
        if current_frame > 0:
            current_frame -= 1

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

print("\nAnnotation program finished.")
print("Labels saved in:", LABEL_FOLDER)
