from PIL import Image
import os
def process_image_channels(image_path):
#将4通道，A通道的统一成三通道的图像；
#  process the 4 channels .png
    print("正在转换图片通道数。。。。。。。")
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name
        # 获取该图片全称
        image = Image.open(img_path)              # 打开特定一张图片
        image = image.resize((64, 64))            # 设置需要转换的图片大小
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
            os.remove(img_path)
            # 用新生成的3通道的图片代替原来的；
            image.save(img_path)
            print("这是个四通道的，处理完了！")
        #  process the 1 channel image
        elif image.mode != 'RGB':
            image = image.convert("RGBA")
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
            os.remove(img_path)
            image.save(img_path)
            print("这是个A通道的，处理完了！")
    print("-----------通道数变换完毕-----------")

def image_reshape(image_path, size):
    i = 1
    print("正在统一图片尺寸。。。。。。。")
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name    # 获取该图片全称
        image = Image.open(img_path)              # 打开特定一张图片
        image = image.resize(size)            # 设置需要转换的图片大小
        os.remove(img_path)
        image.save(img_path)
        print("-----------尺寸统一完毕-----------",i)
        i += 1

image_path = 'img/飞机'
process_image_channels(image_path)
image_reshape(image_path, (100, 100))