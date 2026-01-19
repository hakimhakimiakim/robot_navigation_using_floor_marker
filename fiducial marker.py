import cv2
import cv2.aruco
import os

# ============================
# CONFIG
# ============================
BASE_PATH = r"C:\Users\nazmi\OneDrive\Desktop\JANG\Case Study\opencv_project"
folders = ["Training - Marker", "Testing - Marker"]

# Mapping ID → Area
area_map = {
    1: "Loading Bay",
    2: "Storage Area",
    3: "Exit Gate",
    4: "Charging Station"
}

# Guna dictionary yang sama masa generate marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# ============================
# LOOP FOLDER
# ============================
for folder in folders:
    IN_FOLDER = os.path.join(BASE_PATH, folder)

    for root, _, files in os.walk(IN_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Failed to read {img_path}")
                    continue

                # Convert ke grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect marker
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                if ids is not None:
                    # Lukis marker detection
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)

                    # Print ID marker + Area Mapping
                    for marker_id in ids:
                        area = area_map.get(marker_id[0], "Unknown Area")
                        print(f"{folder} → {file} → Detected Marker ID: {marker_id[0]} → {area}")

                        cv2.putText(img, f"ID: {marker_id[0]} ({area})", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print(f"{folder} → {file} → ❌ No marker detected")

                # Resize untuk paparan (640x480)
                resized = cv2.resize(img, (640, 480))
                cv2.imshow(f"{folder} - {file}", resized)

                # Tunggu ESC untuk keluar ke imej seterusnya
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        cv2.destroyAllWindows()
                        break
