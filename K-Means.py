import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def quantize_colors_kmeans(image_path, n_colors, output_path):
    """
    使用 K-Means 聚類壓縮圖像的色彩空間。
    
    :param image_path: 輸入圖像路徑
    :param n_colors: 壓縮後的顏色數量 (K 值)
    :param output_path: 如果指定，保存量化後的圖像
    :return: 量化後的圖像
    """
    # 加載圖像並轉換為數組格式
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默認為 BGR，轉換為 RGB
    pixels = image_rgb.reshape(-1, 3)  # 壓平成 (像素數, 3) 的數組
    
    # 使用 K-Means 聚類進行顏色壓縮
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    cluster_centers = np.round(kmeans.cluster_centers_).astype("uint8")  # 獲得聚類中心（量化顏色）
    labels = kmeans.labels_  # 獲得每個像素的標籤（所屬聚類）
    
    # 將像素映射為量化顏色
    quantized_pixels = cluster_centers[labels]
    quantized_image = quantized_pixels.reshape(image_rgb.shape)  # 恢復圖像形狀
    
    # 使用 PIL 顯示或保存量化後的圖像
    quantized_image_pil = Image.fromarray(quantized_image)
    if output_path:
        quantized_image_pil.save(output_path)
    return quantized_image_pil

# 測試
image_path = r"C:\Users\Him02\Desktop\Color Quantize\K-Means\RGS04044.JPG"  #圖像路徑
output_path = r"C:\Users\Him02\Desktop\Color Quantize\K-Means\Result Image\output_image.jpg"  # 完成後的圖像路徑
n_colors = 16  # 壓縮到 n 種顏色

quantized_image = quantize_colors_kmeans(image_path, n_colors, output_path)
#quantized_image.show()
