import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation and Thresholding")
        self.image = None

        # Кнопки для загрузки изображения и запуска анализа
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        self.bernsen_button = tk.Button(root, text="Bernsen Threshold", command=self.apply_bernsen, state=tk.DISABLED)
        self.bernsen_button.pack(pady=5)

        self.niblack_button = tk.Button(root, text="Niblack Threshold", command=self.apply_niblack, state=tk.DISABLED)
        self.niblack_button.pack(pady=5)

        self.points_button = tk.Button(root, text="Detect Points", command=self.detect_points, state=tk.DISABLED)
        self.points_button.pack(pady=5)

        self.lines_button = tk.Button(root, text="Detect 45° Lines", command=self.detect_lines_45, state=tk.DISABLED)
        self.lines_button.pack(pady=5)

        self.gradient_button = tk.Button(root, text="Gradient Detection", command=self.detect_gradient,
                                         state=tk.DISABLED)
        self.gradient_button.pack(pady=5)

        # Поле для отображения графика
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                messagebox.showerror("Error", "Unable to load image.")
                return
            self.show_image(self.image, "Loaded Image")
            # Активируем кнопки для анализа
            self.bernsen_button.config(state=tk.NORMAL)
            self.niblack_button.config(state=tk.NORMAL)
            self.points_button.config(state=tk.NORMAL)
            self.lines_button.config(state=tk.NORMAL)
            self.gradient_button.config(state=tk.NORMAL)

    def show_image(self, img, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis("off")
        self.canvas.draw()

    def apply_bernsen(self):
        if self.image is None:
            return
        result = self.bernsen_threshold(self.image)
        self.show_image(result, "Bernsen Threshold")

    def apply_niblack(self):
        if self.image is None:
            return
        result = self.niblack_threshold(self.image)
        self.show_image(result, "Niblack Threshold")

    def detect_points(self):
        if self.image is None:
            return
        result = self.point_detection(self.image)
        self.show_image(result, "Point Detection")

    def detect_lines_45(self):
        if self.image is None:
            return
        result = self.line_detection_45(self.image)
        self.show_image(result, "45° Line Detection")

    def detect_gradient(self):
        if self.image is None:
            return
        result = self.gradient_detection(self.image)
        self.show_image(result, "Gradient Detection")

    def bernsen_threshold(self, image, window_size=15, contrast_threshold=15):
        half_size = window_size // 2
        output = np.zeros(image.shape, dtype=np.uint8)

        for i in range(half_size, image.shape[0] - half_size):
            for j in range(half_size, image.shape[1] - half_size):
                local_region = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
                max_val = int(np.max(local_region))
                min_val = int(np.min(local_region))
                contrast = max_val - min_val
                mid_gray = (max_val + min_val) // 2

                if contrast < contrast_threshold:
                    output[i, j] = 255 if mid_gray > 127 else 0
                else:
                    output[i, j] = 255 if image[i, j] > mid_gray else 0
        return output

    def niblack_threshold(self, image, window_size=15, k=-0.2):
        mean = cv2.blur(image, (window_size, window_size))
        sqmean = cv2.blur(image ** 2, (window_size, window_size))
        stddev = np.sqrt(sqmean - mean ** 2)

        threshold = mean + k * stddev
        output = np.where(image > threshold, 255, 0).astype(np.uint8)
        return output

    def point_detection(self, image):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def line_detection_45(self, image):
        kernel = np.array([[2, -1, -1],
                           [-1, 2, -1],
                           [-1, -1, 2]])
        return cv2.filter2D(image, -1, kernel)

    def gradient_detection(self, image, method="sobel"):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        return np.uint8(gradient_magnitude)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
