import csv
import math

input_file = "position_history.csv"
output_file = "footwork_analysis.csv"

positions = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        frame = int(row["Frame"])
        x = int(row["X_Position"])
        y = int(row["Y_Position"])
        positions.append((frame, x, y))

if len(positions) < 2:
    print("Not enough position data!")
    exit()

movement_events = 0
left_right_events = 0
forward_backward_events = 0
fast_movement_events = 0

threshold = 20
fast_threshold = 80

for i in range(1, len(positions)):

    frame1, x1, y1 = positions[i - 1]
    frame2, x2, y2 = positions[i]

    dx = x2 - x1
    dy = y2 - y1

    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance >= threshold:
        movement_events += 1

    if abs(dx) >= threshold:
        left_right_events += 1

    if abs(dy) >= threshold:
        forward_backward_events += 1

    if distance >= fast_threshold:
        fast_movement_events += 1

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Total Movement Events",
        "Left Right Events",
        "Forward Backward Events",
        "Fast Movement Events"
    ])

    writer.writerow([
        movement_events,
        left_right_events,
        forward_backward_events,
        fast_movement_events
    ])

print("FOOTWORK ANALYSIS COMPLETED!")
print("Movement events:", movement_events)
print("Left/right movement events:", left_right_events)
print("Forward/backward movement events:", forward_backward_events)
print("Fast movement events:", fast_movement_events)
print("Output:", output_file)
