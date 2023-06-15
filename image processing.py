import cv2
import numpy as np

image_path = r"C:\Users\zhand\Desktop\Nice images\4562.png"

# 读取图像
image = cv2.imread(image_path)

# 提高图像的清晰度
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 锐化滤波器
sharp_image = cv2.filter2D(image, -1, kernel)

# 显示图像
cv2.imshow("Enhanced Image", sharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
