import numpy as np
from itertools import product

### internal methods
def cut(ulf, m, k, image):
    lrb = ulf+m+2*k
    return(image[ulf[0]:lrb[0], ulf[1]:lrb[1], ulf[2]:lrb[2]])

def getUpperLeftFront(SIZE, m):
    max_coords = SIZE-m
    coords = list([list([m[i] * j for j in np.arange(SIZE[i]//m[i])]) for i in range(len(SIZE))])
    for dim in range(len(SIZE)):
        if coords[dim][-1] != max_coords[dim]:
            coords[dim].append(max_coords[dim])
    output = []
    for x, y, z in product(*coords):
        output.append((x, y, z))
    return output


def slice_up(image, m, k):
    """
    :param image: input 3D image as np array
    :param m: window size as numpy vector (size_x, size_y, size_z)
    :param k: Field of view size as numpy vector (size_x, size_y, size_z)
    :return: dictionnary with image slices
    """
    SIZE = np.array(image.shape)
    im_pad = np.zeros(shape=SIZE + 2 * k)
    im_pad[k[0]:(k[0] + SIZE[0]), k[1]:(k[1] + SIZE[1]), k[2]:(k[2] + SIZE[2])] = image
    print(im_pad.shape)

    ulfs = getUpperLeftFront(SIZE, m)
    out = {}
    for ulf in ulfs:
        out[ulf] = cut(np.array(ulf), m, k, im_pad)
    return out

def build_up(slices, m, SIZE):
    """
    :param slices: dictionnary with gradients. Same format as output of 'slice_up'
    :param m: window size as numpy vector (size_x, size_y, size_z)
    :param SIZE: 3D image size as numpy vector (size_x, size_y, size_z)
    :return: 3D image of size SIZE
    """
    res = np.zeros(shape=SIZE)
    for ulf, im in slices.items():
        res[ulf[0]:(ulf[0]+m[0]), ulf[1]:(ulf[1]+m[1]), ulf[2]:(ulf[2]+m[2])] = im
    return res

def valid_gradient(slices, k, m):
    """
    :param slices: dictionnnary with slices. Same format as output of 'slice_up'
    :param k: Field of view size as numpy vector (size_x, size_y, size_z)
    :param m: window size as numpy vector (size_x, size_y, size_z)
    :return: The valid part of the gradient map
    """
    res = {}
    for ulf, im in slices.items():
        res[ulf]=im[k[0]:(m[0]+k[0]), k[1]:(m[1]+k[1]), k[2]:(m[2]+k[2])]
    return res

if __name__== '__main__':
    SIZE_out = np.array([500, 500, 500])
    image_out = np.random.normal(size=SIZE_out)
    m_out = np.array([128, 128, 128])
    k_out = np.array([32, 32, 32])
    sliced = slice_up(image_out, m_out, k_out)
    gradients = valid_gradient(sliced, k_out, m_out)
    image_recon = build_up(gradients, m_out, SIZE_out)
    misfit= np.sum(np.square(image_out - image_recon))
    assert misfit == 0.0
