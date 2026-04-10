import os
import shutil

input_folder = "frames_all"
output_folder = "frames_16"

os.makedirs(output_folder, exist_ok=True)

files = sorted(os.listdir(input_folder))

# chọn 16 frame đầu
selected = files[:16]

for file in selected:
    src = os.path.join(input_folder, file)
    dst = os.path.join(output_folder, file)

    shutil.copy(src, dst)

print("Đã lấy 16 frame")