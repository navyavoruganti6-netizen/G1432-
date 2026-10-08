import csv

with open("footwork_analysis.csv", "r") as file:
    reader = csv.DictReader(file)
    row = next(reader)

total = int(row["Total Movement Events"])
left_right = int(row["Left Right Events"])
forward_backward = int(row["Forward Backward Events"])
fast = int(row["Fast Movement Events"])

with open("performance_statistics.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Value",
        "Percentage"
    ])

    writer.writerow([
        "Total Movement Events",
        total,
        "100%"
    ])

    writer.writerow([
        "Left/Right Movement",
        left_right,
        f"{left_right / total * 100:.2f}%"
    ])

    writer.writerow([
        "Forward/Backward Movement",
        forward_backward,
        f"{forward_backward / total * 100:.2f}%"
    ])

    writer.writerow([
        "Fast Movement Events",
        fast,
        f"{fast / total * 100:.2f}%"
    ])

print("PERFORMANCE STATISTICS COMPLETED!")
print("Output: performance_statistics.csv")
