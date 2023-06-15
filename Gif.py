from PIL import Image
import glob

# 设置图像路径和输出 GIF 文件路径
image_path = r'D:\work\pythonProject1\FR\*.png'
output_gif_path = r'D:\work\pythonProject1\animation.gif'

# 读取图像列表
image_files = glob.glob(image_path)
image_files.sort()  # 确保图像按照顺序加载

# 创建一个图像序列列表
image_sequence = []
for image_file in image_files:
    image = Image.open(image_file)
    image_sequence.append(image)

num_images = len(image_files)
print(f"转换为 GIF 的图像数量: {num_images}")
# 将图像序列保存为 GIF 动画
image_sequence[0].save(output_gif_path, save_all=True, append_images=image_sequence[1:], optimize=False, duration=1, loop=0)
print("GIF 动画生成成功！")


