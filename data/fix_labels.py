import os

print("Fixing labels safely...")

base_path = os.path.dirname(__file__)

folders = [
    os.path.join(base_path, "train", "labels"),
    os.path.join(base_path, "val", "labels")
]

# TEXT → NEW CLASS
text_mapping = {
    "driver_with_helmet": 0,
    "passenger_with_helemt": 0,
    "passenger_with_helmet": 0,

    "driver_without_helmet": 1,
    "passenger_without_helemt": 1,
    "passenger_without_helmet": 1
}

# NUMERIC → NEW CLASS
num_mapping = {
    0: 0,
    3: 0,
    5: 1,
    6: 1
}

for folder in folders:
    print(f"\nProcessing: {folder}")

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            first = parts[0]

            # CASE 1: numeric label
            if first.isdigit():
                cls = int(first)

                if cls in num_mapping:
                    parts[0] = str(num_mapping[cls])
                    new_lines.append(" ".join(parts))

            # CASE 2: text label
            else:
                class_name = first

                if class_name in text_mapping:
                    new_cls = text_mapping[class_name]
                    new_line = " ".join([str(new_cls)] + parts[1:])
                    new_lines.append(new_line)

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("\nDone! Labels cleaned ✅")