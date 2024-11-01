import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor")
        self.master.geometry("1200x800")

        self.canvas = Canvas(self.master)
        self.scrollbar = Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.label = Label(self.scrollable_frame, text="Choose an image to process:")
        self.label.pack(pady=10)

        self.choose_button = Button(self.scrollable_frame, text="Choose Image", command=self.choose_image)
        self.choose_button.pack()

        self.original_image_label = Label(self.scrollable_frame)
        self.original_image_label.pack(pady=10)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            7,
            0
        )

        # Local Otsu thresholding
        otsu_segmented = self.local_otsu(image, block_size=10)

        # Point Detection
        points_method1, points_method2 = self.detect_points(image)

        # Line Detection
        lines_method1, lines_method2 = self.detect_lines(image)

        # Edge Detection
        edges_method1, edges_method2 = self.detect_edges(image)

        self.show_images(
            image,
            adaptive_thresh,
            otsu_segmented,
            points_method1,
            points_method2,
            lines_method1,
            lines_method2,
            edges_method1,
            edges_method2
        )

    def local_otsu(self, image, block_size):
        height, width = image.shape
        segmented_image = np.zeros_like(image)

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = image[y:y + block_size, x:x + block_size]
                if block.size == block_size * block_size:
                    _, local_thresh = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    segmented_image[y:y + block_size, x:x + block_size] = local_thresh

        return segmented_image

    def detect_points(self, image):
        # Local Thresholding
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        _, points_method1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Harris Corner Detection
        harris_corners = cv2.cornerHarris(image, 2, 3, 0.04)
        harris_points = cv2.dilate(harris_corners, None)
        points_method2 = np.uint8(harris_points > 0.01 * harris_points.max()) * 255

        return points_method1, points_method2

    def detect_lines(self, image):
        # Adaptive Threshold
        adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        lines_method1 = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

        # Canny Edge
        lines_method2 = cv2.Canny(image, 100, 200)

        return lines_method1, lines_method2


    def detect_edges(self, image):
        # Sobel
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges_method1 = cv2.magnitude(sobelx, sobely).astype(np.uint8)
        _, edges_method1 = cv2.threshold(edges_method1, 50, 255, cv2.THRESH_BINARY)

        # Prewitt
        kernel_prewitt_x = np.array([[1, 0, -1],
                                     [1, 0, -1],
                                     [1, 0, -1]], dtype=np.float32)
        kernel_prewitt_y = np.array([[1, 1, 1],
                                     [0, 0, 0],
                                     [-1, -1, -1]], dtype=np.float32)

        prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_prewitt_x)
        prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_prewitt_y)

        prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)

        _, edges_method2 = cv2.threshold(prewitt_combined, 50, 255, cv2.THRESH_BINARY)

        return edges_method1, edges_method2

    def show_images(self, original, adaptive_thresh, otsu_segmented, points_method1, points_method2,
                    lines_method1, lines_method2, edges_method1, edges_method2):
        display_width = self.master.winfo_width() - 100

        def resize_image_with_aspect(image_array, scale_factor=0.5):
            h, w = image_array.shape
            scale = display_width / w
            new_height = int(h * scale * scale_factor)
            resized_image = Image.fromarray(image_array).resize(
                (int(display_width * scale_factor), new_height), Image.LANCZOS
            )
            return ImageTk.PhotoImage(resized_image)

        # Display the original image in the center
        original_resized = resize_image_with_aspect(original, 0.6)
        original_label = Label(self.scrollable_frame, image=original_resized, text="Original Image", compound="top")
        original_label.image = original_resized  # Keep reference
        original_label.pack(pady=10)

        # Prepare the remaining images for grouping
        images = [
            ("Adaptive Thresholding", adaptive_thresh),
            ("Local Otsu Thresholding", otsu_segmented),
            ("Points Detection - Method 1", points_method1),
            ("Points Detection - Method 2", points_method2),
            ("Lines Detection - Method 1", lines_method1),
            ("Lines Detection - Method 2", lines_method2),
            ("Edges Detection - Method 1", edges_method1),
            ("Edges Detection - Method 2", edges_method2),
        ]

        # Create frames for grouping images by two
        for i in range(0, len(images), 2):
            frame = Frame(self.scrollable_frame)
            frame.pack(pady=10)

            for j in range(2):
                if i + j < len(images):
                    title, img = images[i + j]
                    img_resized = resize_image_with_aspect(img, 0.45)
                    img_label = Label(frame, image=img_resized, text=title, compound="top")
                    img_label.image = img_resized  # Keep reference
                    img_label.pack(side="left", padx=5)


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
