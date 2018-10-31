import numpy as np
from PIL import Image


class Picture():
    def __init__(self, image_source=None, image_array=None):
        """
        圖片矩陣處理
        :param image_source:string 圖片路徑
        :param image_array: 2D numpy array (list of lists) 圖片2D矩陣
        """
        self.image_source = image_source
        self.image_np_array = image_array

    def pic2array(self):
        pic = Image.open(self.image_source).convert('LA')
        width, height = pic.size
        pic_origin_list = np.array(pic)
        pic_gray_list = [[] for i in range(height)]
        for h in range(0,height):
            for w in range(0,width):
                pic_gray_list[h].append(pic_origin_list[h, w, 0]/255)

        return pic_gray_list

    def array2pic(self):
        height = len(self.image_np_array)
        width = len(self.image_np_array[0])
        array4pic = [[] for i in range(height)]
        for h in range(height):
            for w in range(width):
                array4pic[h].append([self.image_np_array[h][w]*255, 255])

        array4pic = np.array(array4pic)
        pic = Image.fromarray(array4pic, 'LA')
        pic.show()
        return pic
