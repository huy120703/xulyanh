import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
from keras.datasets import mnist

# Load dữ liệu MNIST và huấn luyện mô hình
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_feature = [hog(x, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2") for x in x_train]
x_test_feature = [hog(x, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2") for x in x_test]

x_train_feature = np.array(x_train_feature, dtype=np.float32)
x_test_feature = np.array(x_test_feature, dtype=np.float32)

model = LinearSVC(C=10)
model.fit(x_train_feature, y_train)

y_pred = model.predict(x_test_feature)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Hàm xử lý và dự đoán chữ số từ ảnh
def process_and_predict(image_path):
    image = cv2.imread(image_path)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
    _, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)

    # Tìm các vùng chứa chữ số
    contours, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = thre[y:y + h, x:x + w]
        roi = np.pad(roi, ((20, 20), (20, 20)), 'constant', constant_values=0)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Tính toán đặc trưng HOG
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")
        nbr = model.predict(np.array([roi_hog_fd], np.float32))

        # Hiển thị kết quả dự đoán trên ảnh
        cv2.putText(image, str(int(nbr[0])), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("ket qua du doan", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hàm mở file ảnh
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
    if file_path:
        process_and_predict(file_path)

# Hàm tạo giao diện đánh giá
def create_evaluation_window():
    eval_window = tk.Toplevel(root)
    eval_window.title("Đánh giá mô hình")
    eval_window.geometry("800x600")

    bg_image = Image.open("background.png")
    bg_image = bg_image.resize((800, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(eval_window, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    title = tk.Label(eval_window, text="Đánh giá mô hình SVM", font=("Arial", 18, "bold"), bg="#FFFFFF")
    title.pack(pady=10)

    acc_label = tk.Label(eval_window, text=f"Độ chính xác: {accuracy:.2%}", font=("Arial", 14), bg="#FFFFFF")
    acc_label.pack(pady=5)

    cm_title = tk.Label(eval_window, text="Ma trận nhầm lẫn:", font=("Arial", 14), bg="#FFFFFF")
    cm_title.pack(pady=5)

    cm_frame = tk.Frame(eval_window, bg="#FFFFFF")
    cm_frame.pack(pady=10)

    tree = ttk.Treeview(cm_frame, columns=[str(i) for i in range(10)], show="headings", height=10)
    for i in range(10):
        tree.heading(str(i), text=str(i))
        tree.column(str(i), width=50, anchor="center")

    for i, row in enumerate(conf_matrix):
        tree.insert("", "end", values=[str(x) for x in row])

    tree.pack()

    close_button = tk.Button(eval_window, text="Đóng", command=eval_window.destroy, font=("Arial", 12))
    close_button.pack(pady=10)

# Giao diện chính
root = tk.Tk()
root.title("Chương trình nhận dạng chữ số viết tay")
root.geometry("800x600")

bg_image = Image.open("backgr3.png")
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.image = bg_photo
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

title = tk.Label(root, text="Chào mừng đến với chương trình nhận dạng chữ số", font=("Arial", 18, "bold"), bg="#FFFFFF")
title.pack(pady=20)

pred_button = tk.Button(root, text="Dự đoán chữ số", command=open_file, font=("Arial", 14), bg="#FFD700")
pred_button.pack(pady=10)

eval_button = tk.Button(root, text="Đánh giá mô hình", command=create_evaluation_window, font=("Arial", 14), bg="#FFD700")
eval_button.pack(pady=10)

exit_button = tk.Button(root, text="Thoát", command=root.quit, font=("Arial", 14), bg="#FFD700")
exit_button.pack(pady=20)

root.mainloop()
