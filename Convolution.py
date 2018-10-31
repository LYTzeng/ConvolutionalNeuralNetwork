from PicProcessing import Picture
import numpy as np
from FakeData import Fake


class Convolution():
    def __init__(self, image_source=None, filter=None, max_pooling=2):
        """
        CNN 捲積神經網路
        :param image_source: string 圖片路徑
        :param filter: list of lists (2D) 用來做 Featured Map
        """
        if image_source is None:
            self.image_source_dir = 'D:/NTUT/智慧型代理人/作業/CNN/test_data/pic_1.jpg'
        else:
            self.image_source_dir = image_source
        self.picture = Picture(image_source=self.image_source_dir)
        self.image_source_arr = self.picture.pic2array()
        self.image_np = np.array(self.image_source_arr)
        print(self.image_np)
        if filter is None:
            self.filter = [[1,0,-1],[2,0,-2],[1,0,-1]]
        else:
            self.filter = filter
        self.filter = np.array(self.filter)
        self.map = None
        self.map_reLu = None
        self.max_pooling_size = max_pooling

        # 測資
        self.fake = Fake()

    def feature_map(self,src=None):
        """
        依據filter和原始影像陣列計算出Featured Map
        :param src: numpy array
        :return: numpy array
        """
        if src is None:
            height, width = self.size(self.image_np)
            image = self.image_np
        else:
            height, width = self.size(src)
            image = src

        filter_height,filter_width  = self.size(self.filter)
        self.map_height = height - (filter_height - 1)
        self.map_width = width - (filter_width - 1)
        self.map = [[] for i in range(self.map_height)]

        for h in range(0, self.map_height):
            for w in range(0, self.map_width):
                part = image[h:filter_height+h, w:filter_width+w]
                try:
                    self.multi = np.multiply(part, self.filter)
                except Exception as e:
                    print(e)
                map_element = np.sum(self.multi)
                self.map[h].append(map_element)
        return np.array(self.map)

    def reLu(self, src):
        """
        ReLu函數
        :param map: list, featured map
        :return:relu np array
        """
        map_np = src
        self.map_reLu = [[] for i in range(self.map_height)]
        for h in range(self.map_height):
            for w in range(self.map_width):
                if map_np[h, w] < 0 :
                    self.map_reLu[h].append(0)
                else:
                    self.map_reLu[h].append(map_np[h, w])
        reLu_res = np.array(self.map_reLu)
        return reLu_res

    def max_pooling(self, src):
        """
        Max pooling
        :param src: numpy array, reLu結果
        :return: numpy array, max pooling 結果
        """
        height, width = self.size(src)
        res_size = int(height / self.max_pooling_size)
        if height % self.max_pooling_size != 0:
            res_size = res_size + 1
        res = [[] for i in range(res_size)]
        for h in range(0, res_size):
            for w in range(0, res_size):
                part = src[h*self.max_pooling_size:h*self.max_pooling_size+self.max_pooling_size,
                       w*self.max_pooling_size:w*self.max_pooling_size+self.max_pooling_size]
                try:
                    res[h].append(part.max())
                except:
                    pass
        return np.array(res)

    def export(self):
        step1_fm = self.feature_map()
        step1_relu = self.reLu(step1_fm)
        step1_pool = self.max_pooling(step1_relu)
        step2_fm = self.feature_map(step1_pool)
        step2_relu = self.reLu(step2_fm)
        step2_pool = self.max_pooling(step2_relu)

        pic = Picture(image_array = step2_pool)
        return pic.array2pic()

    def size(self, array):
        """
        :param array: list
        :return: height, width
        """
        return len(array),len(array[0])

    def test(self):
        """小測試"""
        self.image_np  = np.array(self.fake.arr3x3)
        self.filter = np.array(self.fake.arr2x2filter)
        print(self.max_pooling(self.reLu(self.feature_map())))
