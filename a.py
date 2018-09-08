import base64
import json
from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re
import sys


class DataAugment(object):
    def __init__(self, image_id=1, form='png'):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.id = image_id
        self.format = '.' + form
        img = cv.imread(str(self.id)+self.format)
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+self.format)
            sys.exit(0)
        self.src = img
        dst1 = cv.flip(img, 0, dst=None)
        dst2 = cv.flip(img, 1, dst=None)
        dst3 = cv.flip(img, -1, dst=None)
        self.flip_x = dst1
        self.flip_y = dst2
        self.flip_x_y = dst3
        cv.imwrite(str(self.id)+'_flip_x'+self.format, self.flip_x)
        cv.imwrite(str(self.id)+'_flip_y'+self.format, self.flip_y)
        cv.imwrite(str(self.id)+'_flip_x_y'+self.format, self.flip_x_y)

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(str(self.id)+'_Gaussian'+self.format, dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Gaussian'+self.format, dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Gaussian'+self.format, dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Gaussian'+self.format, dst4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.5
            increase = 1.4
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
            dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
            dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
            dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
            dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            cv.imwrite(str(self.id)+'_ReduceEp'+self.format, dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_ReduceEp'+self.format, dst3)
            cv.imwrite(str(self.id)+'_flip_y'+'_ReduceEp'+self.format, dst5)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_ReduceEp'+self.format, dst7)
            cv.imwrite(str(self.id)+'_IncreaseEp'+self.format, dst2)
            cv.imwrite(str(self.id)+'_flip_x'+'_IncreaseEp'+self.format, dst4)
            cv.imwrite(str(self.id)+'_flip_y'+'_IncreaseEp'+self.format, dst6)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_IncreaseEp'+self.format, dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.005
            dst1 = self.src
            dst2 = self.flip_x
            dst3 = self.flip_y
            dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    dst2[rand_x, rand_y] = 0
                    dst3[rand_x, rand_y] = 0
                    dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    dst2[rand_x, rand_y] = 255
                    dst3[rand_x, rand_y] = 255
                    dst4[rand_x, rand_y] = 255
            cv.imwrite(str(self.id)+'_Salt'+self.format, dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Salt'+self.format, dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Salt'+self.format, dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Salt'+self.format, dst4)

    def json_generation(self):
        image_names = [str(self.id)+'_flip_x', str(self.id)+'_flip_y', str(self.id)+'_flip_x_y']
        if self.gaussianBlur:
            image_names.append(str(self.id)+'_Gaussian')
            image_names.append(str(self.id)+'_flip_x'+'_Gaussian')
            image_names.append(str(self.id)+'_flip_y' + '_Gaussian')
            image_names.append(str(self.id)+'_flip_x_y'+'_Gaussian')
        if self.changeExposure:
            image_names.append(str(self.id)+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_y'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_IncreaseEp')
        if self.add_saltNoise:
            image_names.append(str(self.id)+'_Salt')
            image_names.append(str(self.id)+'_flip_x' + '_Salt')
            image_names.append(str(self.id)+'_flip_y' + '_Salt')
            image_names.append(str(self.id)+'_flip_x_y' + '_Salt')
        for image_name in image_names:
            with open(image_name+self.format, "rb")as b64:
                base64_data_original = str(base64.b64encode(b64.read()))
                # In pycharm:
                # match_pattern=re.compile(r'b\'(.*)\'')
                # base64_data=match_pattern.match(base64_data_original).group(1)
                # In terminal:
                base64_data = base64_data_original
            with open(str(self.id)+".json", 'r')as js:
                json_data = json.load(js)
                img = utils.img_b64_to_arr(json_data['imageData'])
                height, width = img.shape[:2]
                shapes = json_data['shapes']
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        match_pattern2 = re.compile(r'(.*)_x(.*)')
                        match_pattern3 = re.compile(r'(.*)_y(.*)')
                        match_pattern4 = re.compile(r'(.*)_x_y(.*)')
                        if match_pattern4.match(image_name):
                            point[0] = width - point[0]
                            point[1] = height - point[1]
                        elif match_pattern3.match(image_name):
                            point[0] = width - point[0]
                            point[1] = point[1]
                        elif match_pattern2.match(image_name):
                            point[0] = point[0]
                            point[1] = height - point[1]
                        else:
                            point[0] = point[0]
                            point[1] = point[1]
                json_data['imagePath'] = image_name+self.format
                json_data['imageData'] = base64_data
                json.dump(json_data, open("./"+image_name+".json", 'w'), indent=2)


def usage_display():
    print('***************************************************')
    print('Usage Display:')
    print('1.$ python a.py (default png format)')
    print('2.$ python a.py jpg or $ python a.py png')
    print('3.$ python a.py 1 10  (default png format)')
    print('4.$ python a.py 1 10 jpg or $ python a.py 1 10 png')
    print('***************************************************')


if __name__ == "__main__":
    usage_display()
    form = 'png'
    start_id = 1
    end_id = 1
    if len(sys.argv) == 1:
        pass
    elif len(sys.argv) == 2:
        form = sys.argv[1]
    elif len(sys.argv) == 3:
        start_id = int(sys.argv[1])
        end_id = int(sys.argv[2])
    elif len(sys.argv) == 4:
        start_id = int(sys.argv[1])
        end_id = int(sys.argv[2])
        form = sys.argv[3]
    else:
        print('Parameter error')
        sys.exit()
    for ids in range(start_id, end_id+1, 1):
        dataAugmentObject = DataAugment(ids, form)
        dataAugmentObject.gaussian_blur_fun()
        dataAugmentObject.change_exposure_fun()
        dataAugmentObject.add_salt_noise()
        dataAugmentObject.json_generation()
