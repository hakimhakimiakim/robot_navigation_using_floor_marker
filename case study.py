import cv2
import cv2.aruco as aruco
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
BASE_PATH = r"C:\Users\nazmi\OneDrive\Desktop\JANG\Case Study\opencv_project"
LINE_FOLDER = os.path.join(BASE_PATH, "Testing - Line")
MARKER_FOLDER = os.path.join(BASE_PATH, "Testing - Marker")
MEMORY_PATH = os.path.join(BASE_PATH, "Memory")
os.makedirs(MEMORY_PATH, exist_ok=True)
LOG_FILE = os.path.join(MEMORY_PATH, "summary.txt")

area_map = {
    1: "Loading Bay",
    2: "Storage Area",
    3: "Exit Gate",
    4: "Charging Station"
}

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

# ============================
# FEATURE EXTRACTION
# ============================
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=20)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            angles.append(np.degrees(np.arctan2(y2-y1, x2-x1)))

    mean_angle = np.mean(angles) if angles else 0
    edge_density = cv2.countNonZero(edges)/edges.size

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = img.shape[0]*img.shape[1]

    mask_red = cv2.inRange(hsv,(0,100,100),(10,255,255)) + cv2.inRange(hsv,(160,100,100),(179,255,255))
    mask_yellow = cv2.inRange(hsv,(20,100,100),(30,255,255))
    mask_green  = cv2.inRange(hsv,(40,40,40),(70,255,255))
    mask_blue   = cv2.inRange(hsv,(100,150,0),(140,255,255))

    return [
        mean_angle,
        edge_density,
        cv2.countNonZero(mask_red)/total,
        cv2.countNonZero(mask_yellow)/total,
        cv2.countNonZero(mask_green)/total,
        cv2.countNonZero(mask_blue)/total
    ]

# ============================
# BUILD DATASET (os.walk)
# ============================
X, y = [], []

for root, _, files in os.walk(LINE_FOLDER):
    for file in files:
        if not file.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Failed to read:", img_path)
            continue

        X.append(extract_features(img))
        fname = file.lower()
        if "left" in fname:
            y.append("LEFT")
        elif "right" in fname:
            y.append("RIGHT")
        else:
            y.append("STRAIGHT")

if len(X)<2:
    raise ValueError("❌ Dataset terlalu sikit (minimum 2 gambar)")

print("✅ Dataset loaded:", len(X))
print("📊 Label count:", Counter(y))

# ============================
# SAFE kNN NEIGHBORS
# ============================
def safe_n_neighbors(X):
    return min(3, len(X))

knn = KNeighborsClassifier(n_neighbors=safe_n_neighbors(X))
dt  = DecisionTreeClassifier(max_depth=5, random_state=42)

knn.fit(X,y)
dt.fit(X,y)

# ============================
# TESTING + PREPROCESSING POPUP
# ============================
y_true, y_pred_knn, y_pred_dt = [], [], []
log_results = []

for root, _, files in os.walk(LINE_FOLDER):
    for file in files:
        if not file.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        img_path = os.path.join(root,file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur,50,150)
        _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

        g = cv2.cvtColor(cv2.resize(gray,(320,240)), cv2.COLOR_GRAY2BGR)
        b = cv2.cvtColor(cv2.resize(blur,(320,240)), cv2.COLOR_GRAY2BGR)
        e = cv2.cvtColor(cv2.resize(edges,(320,240)), cv2.COLOR_GRAY2BGR)
        bi= cv2.cvtColor(cv2.resize(binary,(320,240)), cv2.COLOR_GRAY2BGR)

        cv2.putText(g,"Grayscale",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(b,"Gaussian Blur",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(e,"Canny Edge",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(bi,"Binary Segmentation",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        cv2.imshow("Preprocessing",np.hstack([g,b,e,bi]))
        cv2.waitKey(0)
        cv2.destroyWindow("Preprocessing")

        feat = extract_features(img)
        pk = knn.predict([feat])[0]
        ck = np.max(knn.predict_proba([feat])[0])*100
        pd = dt.predict([feat])[0]
        cd = np.max(dt.predict_proba([feat])[0])*100

        if "left" in file.lower(): true="LEFT"
        elif "right" in file.lower(): true="RIGHT"
        else: true="STRAIGHT"

        y_true.append(true)
        y_pred_knn.append(pk)
        y_pred_dt.append(pd)

        cv2.putText(img,f"kNN: {pk} ({ck:.1f}%)",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        cv2.putText(img,f"DT : {pd} ({cd:.1f}%)",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,0),2)

        cv2.imshow("Line Classification",cv2.resize(img,(640,480)))
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(MEMORY_PATH,f"Line_{file}"),img)

        log_results.append(f"{file} | True:{true} | kNN:{pk}({ck:.1f}%) | DT:{pd}({cd:.1f}%)")

cv2.destroyAllWindows()

# ============================
# ACCURACY
# ============================
def acc(t,p): return sum(a==b for a,b in zip(t,p))/len(t)*100

acc_knn = acc(y_true,y_pred_knn)
acc_dt  = acc(y_true,y_pred_dt)

print(f"\n✅ kNN Accuracy: {acc_knn:.1f}%")
print(f"✅ DT  Accuracy: {acc_dt:.1f}%")

# ============================
# MARKER DETECTION
# ============================
for root, _, files in os.walk(MARKER_FOLDER):
    for file in files:
        if not file.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        img_path = os.path.join(root,file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
            h,w = img.shape[:2]
            for marker_id in ids:
                area = area_map.get(marker_id[0],"Unknown Area")
                cv2.putText(img,f"ID {marker_id[0]} ({area})",(w-350,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                print(f"✅ Marker {marker_id[0]} -> {area}")
        else:
            cv2.putText(img,"NO MARKER DETECTED",(30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        cv2.imshow("Marker Detection",cv2.resize(img,(640,480)))
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(MEMORY_PATH,f"Marker_{file}"),img)

cv2.destroyAllWindows()

# ============================
# CONFUSION MATRIX
# ============================
labels = ["LEFT","STRAIGHT","RIGHT"]

cm_knn = confusion_matrix(y_true,y_pred_knn,labels=labels)
plt.figure()
sns.heatmap(cm_knn,annot=True,fmt="d",xticklabels=labels,yticklabels=labels)
plt.title("Confusion Matrix - kNN")
plt.show()

cm_dt = confusion_matrix(y_true,y_pred_dt,labels=labels)
plt.figure()
sns.heatmap(cm_dt,annot=True,fmt="d",xticklabels=labels,yticklabels=labels)
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# ============================
# LEARNING CURVE SAFE FUNCTION
# ============================
def plot_learning_curve_safe(model, X, y, title="Learning Curve"):
    if len(X)<2:
        print("⚠️ Dataset terlalu kecil untuk learning curve")
        return
    if isinstance(model, KNeighborsClassifier):
        model.set_params(n_neighbors=min(model.n_neighbors, len(X)))
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=min(3,len(X)), n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5), scoring='accuracy'
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
    plt.plot(train_sizes, val_mean, 'o-', color="green", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================
# PLOT LEARNING CURVES
# ============================
plot_learning_curve_safe(knn, X, y, title="Learning Curve - kNN")
plot_learning_curve_safe(dt, X, y, title="Learning Curve - Decision Tree")

# ============================
# SAVE REPORT
# ============================
with open(LOG_FILE,"w",encoding="utf-8") as f:
    f.write("\n".join(log_results))
    f.write("\n\n=== SUMMARY ===\n")
    f.write(f"kNN Accuracy: {acc_knn:.1f}%\n")
    f.write(f"DT Accuracy : {acc_dt:.1f}%\n\n")
    f.write("kNN Report:\n")
    f.write(classification_report(y_true,y_pred_knn))
    f.write("\nDT Report:\n")
    f.write(classification_report(y_true,y_pred_dt))

print(f"\n📁 Report saved: {LOG_FILE}")
