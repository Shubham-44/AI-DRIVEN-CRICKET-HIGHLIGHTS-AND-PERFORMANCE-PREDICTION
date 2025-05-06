import os
import shutil
import random

# Configuration
FRAMES_FOLDER = 'data/frames'
ANNOTATIONS_FOLDER = 'data/annotations'
OUTPUT_DIR = 'data/split'
TRAIN_RATIO = 0.8  # 80% training, 20% validation


def prepare_split(frames_folder, annotations_folder, output_dir, train_ratio):
    # Create output folders
    train_images_dir = os.path.join(output_dir, 'images/train')
    val_images_dir = os.path.join(output_dir, 'images/val')
    train_labels_dir = os.path.join(output_dir, 'labels/train')
    val_labels_dir = os.path.join(output_dir, 'labels/val')

    for folder in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(folder, exist_ok=True)

    # List all frame files
    frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    random.shuffle(frame_files)

    split_point = int(len(frame_files) * train_ratio)
    train_files = frame_files[:split_point]
    val_files = frame_files[split_point:]

    print(f"🔢 Total images: {len(frame_files)}")
    print(f"📦 Training images: {len(train_files)}")
    print(f"🧪 Validation images: {len(val_files)}")

    # Move files
    for file_list, image_dest, label_dest in [
        (train_files, train_images_dir, train_labels_dir),
        (val_files, val_images_dir, val_labels_dir)
    ]:
        for img_file in file_list:
            img_src_path = os.path.join(frames_folder, img_file)
            label_src_path = os.path.join(annotations_folder, img_file.replace('.jpg', '.txt'))

            if os.path.exists(img_src_path) and os.path.exists(label_src_path):
                shutil.copy(img_src_path, image_dest)
                shutil.copy(label_src_path, label_dest)
            else:
                print(f"⚠️ Missing pair for: {img_file}")

    print("✅ Dataset split completed successfully!")


if __name__ == "__main__":
    prepare_split(FRAMES_FOLDER, ANNOTATIONS_FOLDER, OUTPUT_DIR, TRAIN_RATIO)
