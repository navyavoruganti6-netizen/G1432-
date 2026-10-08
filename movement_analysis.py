import csv
import math

input_file = "position_history.csv"
output_file = "movement_analysis.csv"

positions = []

with open(input_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x = int(row["X_Position"])
        y = int(row["Y_Position"])
        positions.append((x, y))

if len(positions) < 2:
    print("Not enough position data!")
    exit()

total_distance = 0
horizontal_distance = 0
vertical_distance = 0
max_step = 0

for i in range(1, len(positions)):

    x1, y1 = positions[i - 1]
    x2, y2 = positions[i]

    dx = x2 - x1
    dy = y2 - y1

    distance = math.sqrt(dx ** 2 + dy ** 2)

    total_distance += distance
    horizontal_distance += abs(dx)
    vertical_distance += abs(dy)

    if distance > max_step:
        max_step = distance

average_distance = total_distance / (len(positions) - 1)

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Total Movement (pixels)",
        "Average Movement Per Frame (pixels)",
        "Horizontal Movement (pixels)",
        "Vertical Movement (pixels)",
        "Maximum Movement Between Frames (pixels)"
    ])

    writer.writerow([
        round(total_distance, 2),
        round(average_distance, 2),
        round(horizontal_distance, 2),
        round(vertical_distance, 2),
        round(max_step, 2)
    ])

print("MOVEMENT ANALYSIS COMPLETED!")
print("Total movement:", round(total_distance, 2), "pixels")
print("Average movement per frame:", round(average_distance, 2), "pixels")
print("Horizontal movement:", round(horizontal_distance, 2), "pixels")
print("Vertical movement:", round(vertical_distance, 2), "pixels")
print("Maximum movement between frames:", round(max_step, 2), "pixels")
print("Output:", output_file)
