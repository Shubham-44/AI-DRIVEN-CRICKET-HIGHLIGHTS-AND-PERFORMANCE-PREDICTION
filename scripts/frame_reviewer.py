
import cv2
import os

# --- CONFIG ---
FOLDER = 'data/detected_frames'
KEY_ACCEPT = ord('y')  # Press 'y' to keep
KEY_DELETE = ord('n')  # Press 'n' to delete
WINDOW_NAME = 'Frame Reviewer'
# --------------

images = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(('.jpg', '.png'))])

for img_name in images:
    img_path = os.path.join(FOLDER, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    cv2.imshow(WINDOW_NAME, img)
    print(f"🖼️ Reviewing: {img_name} — press 'y' to keep, 'n' to delete...")

    key = cv2.waitKey(0)

    if key == KEY_DELETE:
        os.remove(img_path)
        print(f"❌ Deleted: {img_name}")
    elif key == KEY_ACCEPT:
        print(f"✅ Kept: {img_name}")
    elif key == 27:  # ESC
        print("🛑 Stopping reviewer.")
        break

cv2.destroyAllWindows()
