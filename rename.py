import os

folder = r"D:\datn\archive\dataset\train\NonFight"

for i, f in enumerate(os.listdir(folder)):
    path = os.path.join(folder, f)

    if not f.endswith(".avi"):
        continue

    new_name = f"NonFight_{i}.avi"
    os.rename(path, os.path.join(folder, new_name))

print("Done renaming")