from pathlib import Path

# Use the V5 processing script, but temporarily point it to the 10-second test video.
source = Path("process_v5.py")
target = Path("process_v5_test.py")

text = source.read_text(encoding="utf-8")

text = text.replace(
    'INPUT_VIDEO = Path("video/badminton_test.mp4")',
    'INPUT_VIDEO = Path("video/badminton_test_10sec.mp4")'
)

text = text.replace(
    'badminton_coordinates_v5.csv',
    'badminton_coordinates_v5_test.csv'
)

text = text.replace(
    'badminton_coordinates_v5.json',
    'badminton_coordinates_v5_test.json'
)

target.write_text(text, encoding="utf-8")

print("=" * 60)
print("V5 TEST SCRIPT CREATED")
print("=" * 60)
print(f"Script: {target}")
print(f"Video : video/badminton_test_10sec.mp4")
print("=" * 60)
