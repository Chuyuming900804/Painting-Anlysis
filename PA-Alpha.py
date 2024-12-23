import cv2
import numpy as np
from sklearn.cluster import KMeans
import csv

#檢視各階段結果用
def show_image(title, image, scale=0.5):
    """Display an image in a window, scaled to fit the screen."""
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 1: Preprocessing  # 降噪預處理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    show_image("Denoised Image", denoised)
    return denoised

# Step 2: Segment image into regions  # 影像分區
def segment_image(image):
    # Convert to grayscale and detect edges  # 邊緣檢測
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    show_image("Edges", edges)

     # Detect contours for fine details (fractals)  # 偵測破碎輪廓(目標是頭髮鬍鬚)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fractal_mask = np.zeros_like(gray)
    cv2.drawContours(fractal_mask, contours, -1, (255), thickness=cv2.FILLED)
    show_image("Fractal Mask", fractal_mask)

    
    # Detect flat regions and gradients using color continuity  # 偵測平塗色塊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = cv2.Laplacian(blurred, cv2.CV_64F)
    gradient_mask = np.uint8(np.abs(gradient) > 20) * 255
    show_image("Gradient Mask", gradient_mask)

    return fractal_mask, gradient_mask


# Step 3: Analyze flat regions (K-Means clustering)  # 對平塗區塊進行Kmeans類聚
def analyze_flat_regions(image, mask):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masked_pixels = hsv_image[mask > 0]

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(masked_pixels)

    # Cluster centers represent dominant colors
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    
    # Calculate pixel proportions for each cluster   # 計算各類聚比例
    unique, counts = np.unique(labels, return_counts=True)
    proportions = dict(zip(unique, counts / len(labels)))

    clustered_image = np.zeros_like(hsv_image)
    for i, label in enumerate(labels):
        clustered_image[mask > 0][i] = kmeans.cluster_centers_[label]
    clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_HSV2RGB)
    show_image("Flat Regions Clustering", clustered_image)

    return colors, proportions

# Step 4: Analyze gradient regions  # 漸層區塊分析
def analyze_gradient_regions(image, mask):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masked_pixels = hsv_image[mask > 0]

    # Find maximum saturation for each hue  # 同色相取最大飽和度者
    hues = masked_pixels[:, 0]
    saturations = masked_pixels[:, 1]
    max_saturation_index = np.argmax(saturations)

    dominant_color = masked_pixels[max_saturation_index]

    gradient_image = np.zeros_like(image)
    gradient_image[mask > 0] = dominant_color
    show_image("Gradient Regions", gradient_image)

    return dominant_color

# Step 5: Calculate region proportions #沒看懂GPT在幹嘛
def calculate_proportions(image, masks):
    total_pixels = image.shape[0] * image.shape[1]
    proportions = {name: np.sum(mask > 0) / total_pixels for name, mask in masks.items()}
    return proportions

# Step 6: Export results to CSV 輸出成CSV
def export_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Region Type", "Color/Label", "Proportion"])
        for row in data:
            writer.writerow(row)

# Main pipeline
def main(image_path, output_csv):
    image = preprocess_image(image_path)
    fractal_mask, gradient_mask = segment_image(image)

    # Flat region analysis
    flat_mask = cv2.bitwise_not(fractal_mask + gradient_mask)
    show_image("Flat Mask", flat_mask)
    flat_colors, flat_proportions = analyze_flat_regions(image, flat_mask)

    # Gradient region analysis
    gradient_color = analyze_gradient_regions(image, gradient_mask)

    # Proportions
    masks = {"Fractal": fractal_mask, "Flat": flat_mask, "Gradient": gradient_mask}
    proportions = calculate_proportions(image, masks)

    # Prepare data for CSV
    data = []
    data.append(["Fractal", "-", proportions["Fractal"]])
    for i, (color, proportion) in enumerate(flat_proportions.items()):
        data.append(["Flat", f"Color {i+1}", proportion])
    data.append(["Gradient", gradient_color.tolist(), proportions["Gradient"]])

    # Export
    export_to_csv(output_csv, data)
#"""

# Example usage
image_path = r"C:\Users\Him02\Desktop\Color Quantize\Tester.JPG"
output_csv = r"C:\Users\Him02\Desktop\Color Quantize\PaintingAnalyzer\color_proportions.csv"
main(image_path, output_csv)
