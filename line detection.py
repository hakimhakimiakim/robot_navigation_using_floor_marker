
import cv2
import os
import numpy as np

# ============================
# CONFIG
# ============================
BASE_PATH = r"C:\Users\nazmi\OneDrive\Desktop\JANG\Case Study\opencv_project"
folders = ["Training", "Testing"]

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

                # Step 1: Tukar ke grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Step 2: Pre-processing (blur + edge detection)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)

                # Paparkan Gray dan Edges (popup kecil)
                cv2.imshow("Gray", cv2.resize(gray, (320, 240)))
                cv2.imshow("Edges", cv2.resize(edges, (320, 240)))

                # Step 3: Hough Transform untuk kesan garisan
                lines = cv2.HoughLinesP(edges,
                                        rho=1,
                                        theta=np.pi/180,
                                        threshold=50,
                                        minLineLength=30,
                                        maxLineGap=20)

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Step 4: Color detection (HSV)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Merah
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([179, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

                # Kuning
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Tentukan arah
                if cv2.countNonZero(mask_red) > 0:
                    direction = "TURN LEFT"
                elif cv2.countNonZero(mask_yellow) > 0:
                    direction = "TURN RIGHT"
                else:
                    direction = "STRAIGHT"

                cv2.putText(img, f"GLOBAL: {direction}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Step 5: Resize imej sebelum tunjuk (640x480)
                resized = cv2.resize(img, (640, 480))

                # Step 6: Paparkan hasil akhir
                cv2.imshow(f"{folder} - {file}", resized)

                # Tunggu ESC untuk keluar ke imej seterusnya
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        cv2.destroyAllWindows()
                        break
