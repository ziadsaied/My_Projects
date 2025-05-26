import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats
from PIL import Image, ImageTk 

def point_operations(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    add_img = cv2.add(img, 50)
    sub_img = cv2.subtract(img, 50)
    div_img = cv2.divide(img, 2)
    comp_img = cv2.bitwise_not(img)
    show_results(['Original', 'Addition', 'Subtraction', 'Division', 'Complement'],
                 [img, add_img, sub_img, div_img, comp_img], cmap='gray')

def color_operations(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red_boost = img_rgb.copy()
    red_boost[:, :, 0] = np.clip(red_boost[:, :, 0] + 50, 0, 255)
    swapped = img_rgb.copy()
    swapped[:, :, 1], swapped[:, :, 0] = img_rgb[:, :, 0], img_rgb[:, :, 1]
    no_red = img_rgb.copy()
    no_red[:, :, 0] = 0
    show_results(['Original', 'Red Boost', 'Swap R↔G', 'No Red'],
                 [img_rgb, red_boost, swapped, no_red])

def histogram_operations(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    equalized = cv2.equalizeHist(img)
    show_results(['Original', 'Stretched', 'Equalized'], [img, stretched, equalized], cmap='gray')

def neighborhood_filters(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))  # حجم أصغر لتحسين الأداء
    avg = cv2.blur(img, (3, 3))
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    maxf = cv2.dilate(img, np.ones((3, 3), np.uint8))
    minf = cv2.erode(img, np.ones((3, 3), np.uint8))
    median = cv2.medianBlur(img, 3)
    mode = mode_filter(img)
    show_results(['Original', 'Average', 'Laplacian', 'Max', 'Min', 'Median', 'Mode'],
                 [img, avg, lap, maxf, minf, median, mode], cmap='gray')

def mode_filter(img):
    padded = np.pad(img, ((1,1),(1,1)), mode='constant')
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+3, j:j+3].flatten()
            mode = stats.mode(window, keepdims=True)[0][0]
            output[i, j] = mode
    return output

def add_salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = int(amount * img.size * 0.5)
    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i-1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i-1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_gaussian_noise(img, mean=0, std=25):
    gauss = np.random.normal(mean, std, img.shape).astype('float32')
    noisy = img.astype('float32') + gauss
    return np.clip(noisy, 0, 255).astype('uint8')

def outlier_filter(img):
    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')
    out_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = padded[i:i+3, j:j+3].flatten()
            center = img[i, j]
            median = np.median(win)
            if abs(int(center) - int(median)) > 30:
                out_img[i, j] = median
    return out_img

def image_restoration(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sp_noisy = add_salt_pepper_noise(img)
    sp_avg = cv2.blur(sp_noisy, (3, 3))
    sp_median = cv2.medianBlur(sp_noisy, 3)
    sp_outlier = outlier_filter(sp_noisy)
    gauss_noisy = add_gaussian_noise(img)
    gauss_avg = ((img.astype('float32') + gauss_noisy.astype('float32')) / 2).astype('uint8')
    gauss_filtered = cv2.blur(gauss_noisy, (3, 3))
    show_results(['Original', 'S&P', 'S&P Avg', 'S&P Median', 'S&P Outlier',
                  'Gaussian', 'Gauss Avg', 'Gauss Filter'],
                 [img, sp_noisy, sp_avg, sp_median, sp_outlier,
                  gauss_noisy, gauss_avg, gauss_filtered], cmap='gray')

def segmentation_thresholding(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, basic_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    show_results(['Original', 'Basic Thresh', 'Otsu Thresh', 'Adaptive Thresh'],
                 [img, basic_thresh, otsu_thresh, adaptive_thresh], cmap='gray')

def sobel_edge_detection(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    show_results(['Original', 'Sobel X', 'Sobel Y', 'Sobel Combined'],
                 [img, np.abs(sobelx), np.abs(sobely), sobel_combined], cmap='gray')

def morphology_operations(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel)
    erosion = cv2.erode(binary, kernel)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    internal = cv2.subtract(binary, erosion)
    external = cv2.subtract(dilation, binary)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    show_results(['Binary', 'Dilation', 'Erosion', 'Opening',
                  'Internal', 'External', 'Gradient'],
                 [binary, dilation, erosion, opening,
                  internal, external, gradient], cmap='gray')

def show_results(titles, images, cmap=None):
    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(2, (len(images)+1)//2, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Project")
        self.img_path = None
        self.image_label = None
        self.path_label = None
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        self.path_label = tk.Label(top_frame, text="No image loaded", fg="blue")
        self.path_label.pack()

        preview_frame = tk.Frame(self.root)
        preview_frame.pack(pady=5)

        self.image_label = tk.Label(preview_frame)
        self.image_label.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Load Image", command=self.load_image, bg="lightblue").grid(row=0, column=0, padx=5, pady=5)

        tasks = [
            ("Point Ops", point_operations),
            ("Color Ops", color_operations),
            ("Histogram", histogram_operations),
            ("Filters", neighborhood_filters),
            ("Restoration", image_restoration),
            ("Segmentation", segmentation_thresholding),
            ("Edge Detection", sobel_edge_detection),
            ("Morphology", morphology_operations),
        ]
        for i, (name, func) in enumerate(tasks):
            tk.Button(btn_frame, text=name, width=18,
                      command=lambda f=func: self.run_task(f),
                      bg="#e0e0e0").grid(row=(i // 4) + 1, column=i % 4, padx=5, pady=5)

        tk.Button(self.root, text="Exit", command=self.root.quit, bg="tomato", fg="white").pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img_path = path
            self.path_label.config(text=f"Loaded Image: {path}")
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img)
            self.image_label.image = img

    def run_task(self, func):
        if self.img_path:
            plt.close('all')
            func(self.img_path)
        else:
            self.path_label.config(text="⚠️ Please load an image first", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()