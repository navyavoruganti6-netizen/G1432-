import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import glob
root = tk.Tk()
root.title("Badminton Footwork Annotation")
root.geometry("1400x850")



# =========================
# SETTINGS
# =========================

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
current_index = 0

start_x = start_y = None
rect = None
# =========================
# MAIN WINDOW
# =========================

root = tk.Tk()
root.title("Badminton Footwork Annotation")
root.geometry("1400x850")

# =========================
# TOP INFORMATION
# =========================

title = tk.Label(
    root,
    text="BADMINTON FOOTWORK ANNOTATION",
    font=("Arial", 18, "bold")
)
title.pack(pady=5)

info = tk.Label(
    root,
    text="Draw a box around the PLAYER → choose the correct footwork class",
    font=("Arial", 12)
)
info.pack()

# =========================
# MAIN AREA
# =========================

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Image area
image_frame = tk.Frame(main_frame)
image_frame.pack(side="left", padx=10, pady=10)

canvas = tk.Canvas(
    image_frame,
    width=950,
    height=650,
    bg="gray"
)
canvas.pack()

# =========================
# GUIDE AREA
# =========================

guide_frame = tk.Frame(main_frame, width=350)
guide_frame.pack(side="right", fill="y", padx=10, pady=10)
# =========================
# MAIN WINDOW
# =========================

# =========================
# TOP INFORMATION
# =========================

title = tk.Label(
    root,
    text="BADMINTON FOOTWORK ANNOTATION",
    font=("Arial", 18, "bold")
)
title.pack(pady=5)

info = tk.Label(
    root,
    text="Draw a box around the PLAYER → choose the correct footwork class",
    font=("Arial", 12)
)
info.pack()

# =========================
# MAIN AREA
# =========================

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Image area
image_frame = tk.Frame(main_frame)
image_frame.pack(side="left", padx=10, pady=10)

canvas = tk.Canvas(
    image_frame,
    width=950,
    height=650,
    bg="gray"
)
canvas.pack()
# =========================
# SKIP BUTTON
# =========================

def skip_frame():
    next_frame()

tk.Button(
    guide_frame,
    text="SKIP FRAME",
    font=("Arial", 11, "bold"),
    width=30,
    height=2,
    command=skip_frame
).pack(pady=15)

# =========================
# FRAME INFORMATION
# =========================

frame_info = tk.Label(
    guide_frame,
    text="",
    font=("Arial", 11),
    justify="left"
)
frame_info.pack(pady=10)

# =========================
# DRAW BOX
# =========================

def mouse_down(event):
    global start_x, start_y, rect

    start_x = event.x
    start_y = event.y

    if rect:
        canvas.delete(rect)

    rect = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="red",
        width=3
    )

def mouse_move(event):
    if rect:
        canvas.coords(
            rect,
            start_x,
            start_y,
            event.x,
            event.y
        )

def mouse_up(event):
    pass

canvas.bind("<Button-1>", mouse_down)
canvas.bind("<B1-Motion>", mouse_move)
canvas.bind("<ButtonRelease-1>", mouse_up)
# =========================
# SAVE YOLO ANNOTATION
# =========================

def save_annotation(class_name):

    global start_x, start_y, rect

    if start_x is None or rect is None:
        messagebox.showwarning(
            "No Box",
            "First draw a box around the player."
        )
        return

    coords = canvas.coords(rect)

    if len(coords) != 4:
        return

    x1, y1, x2, y2 = coords

    if x1 > x2:
        x1, x2 = x2, x1

    if y1 > y2:
        y1, y2 = y2, y1

    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()

    img_w, img_h = original_size

    scale = min(
        canvas_w / img_w,
        canvas_h / img_h
    )

    display_w = img_w * scale
    display_h = img_h * scale

    offset_x = (canvas_w - display_w) / 2
    offset_y = (canvas_h - display_h) / 2

    x1_img = (x1 - offset_x) / scale
    y1_img = (y1 - offset_y) / scale
    x2_img = (x2 - offset_x) / scale
    y2_img = (y2 - offset_y) / scale

    box_x = (x1_img + x2_img) / 2
    box_y = (y1_img + y2_img) / 2
    box_w = x2_img - x1_img
    box_h = y2_img - y1_img

    x_center = box_x / img_w
    y_center = box_y / img_h
    
# =========================
# LOAD FRAME
# =========================

def load_frame():

    global photo, original_size, rect
    global start_x, start_y

    if not frames:
        messagebox.showerror(
            "Error",
            "No JPG frames found in footwork_frames."
        )
        return

    image = Image.open(frames[current_index])
    original_size = image.size

    canvas_w = 950
    canvas_h = 650

    scale = min(
        canvas_w / image.width,
        canvas_h / image.height
    )

    new_size = (
        int(image.width * scale),
        int(image.height * scale)
    )

    image = image.resize(new_size)

    display = Image.new(
        "RGB",
        (canvas_w, canvas_h),
        "gray"
    )

    offset_x = (canvas_w - new_size[0]) // 2
    offset_y = (canvas_h - new_size[1]) // 2

    display.paste(image, (offset_x, offset_y))

    photo = ImageTk.PhotoImage(display, master=root)
    canvas.delete("all")

    canvas.create_image(
        0,
        0,
        anchor="nw",
        image=photo
    )
    canvas.image = photo
    rect = None
    start_x = start_y = None

    frame_info.config(
        text=(
            f"Frame: {current_index + 1} / {len(frames)}\n"
            f"File: {os.path.basename(frames[current_index])}\n\n"
            "1. Draw box around player\n"
            "2. Select footwork class\n"
            "3. If unclear → SKIP FRAME"
        )
    )
# =========================
# NEXT / PREVIOUS
# =========================

def next_frame():

    global current_index

    if current_index < len(frames) - 1:
        current_index += 1
        load_frame()
    else:
        messagebox.showinfo(
            "Finished",
            "All frames have been processed!"
        )


def previous_frame():

    global current_index

    if current_index > 0:
        current_index -= 1
        load_frame()


# =========================
# KEYBOARD SHORTCUTS
# =========================

root.bind("1", lambda e: choose_class(CLASSES[0]))
root.bind("2", lambda e: choose_class(CLASSES[1]))
root.bind("3", lambda e: choose_class(CLASSES[2]))
root.bind("4", lambda e: choose_class(CLASSES[3]))
root.bind("5", lambda e: choose_class(CLASSES[4]))
root.bind("6", lambda e: choose_class(CLASSES[5]))

root.bind("n", lambda e: next_frame())
root.bind("p", lambda e: previous_frame())
root.bind("s", lambda e: skip_frame())


# =========================
# START
# =========================

load_frame()

root.mainloop()
