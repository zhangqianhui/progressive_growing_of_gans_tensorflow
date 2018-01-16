import os
import errno
import numpy as np
import scipy
import scipy.misc

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class CelebA(object):

    def __init__(self, image_path):

        self.dataname = "CelebA"
        self.dims = 64*64
        self.shape = [64, 64, 3]
        self.image_size = 64
        self.image_list = read_image_list(category=image_path)

    def load_celebA(self, image_path):

        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image

        return images_list

    @staticmethod
    def getShapeForData(filenames, resize_w=64):
        array = [get_image(batch_file, 128, is_crop=True, resize_w = resize_w,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):

        ro_num = len(self.image_list) / batch_size - 1
        if batch_num % ro_num == 0:

            length = len(self.image_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.image_list = np.array(self.image_list)
            self.image_list = self.image_list[perm]

            print "images shuffle"

        return self.image_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx= 64 , is_crop=False, resize_w= 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h, crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)

    if rate < 0.5:
        x = np.fliplr(x)

    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

    # return scipy.misc.imresize(x[45: 45 + 128, 25:25 + 128], [resize_w, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return ((image + 1.)* 127.5).astype(np.uint8)

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' in file:
            filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames




