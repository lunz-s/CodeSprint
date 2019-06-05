from AdversarialRegularizer import AdversarialRegulariser
import slicing_methods as sl
import numpy as np

class AdversarialSplitting(object):

    def __init__(self, path, IMAGE_SIZE, NETWORK, BATCH_SIZE, s=0.0, cutoff=20.0, gamma=1.0, lmb=10.0):
        self.m = NETWORK.m
        self.k = NETWORK.k
        self.batch_size = BATCH_SIZE

        self.regularizer = AdversarialRegulariser(path, IMAGE_SIZE, NETWORK, s=s, cutoff=cutoff, gamma=gamma, lmb=lmb)

    def reset_batch_size(self, size):
        self.batch_size = size


    def train(self, groundTruth, adversarial, learning_rate):
        ground_slices = sl.slice_up(groundTruth, self.m, self.k)
        adv_slices = sl.slice_up(adversarial, self.m, self.k)
        size = ground_slices[(0,0,0)].shape
        bs = self.batch_size+size

        k = 0
        gt_batch = np.zeros(shape=bs)
        adv_batch = np.zeros(shape=bs)
        for ulf, im_true in ground_slices:
            im_adv = adv_slices(ulf)
            if k<self.batch_size:
                gt_batch[k,...] = im_true
                adv_batch[k,...] = im_adv
                k =+ 1
            else:
                self.regularizer.train(gt_batch, adv_batch, learning_rate)
                k=0

    def test(self, groundTruth, adversarial):
        pass

    def evaluate(self, data):
        pass

    def save(self):
        self.regularizer.save()

    def load(self):
        self.regularizer.load()

    def end(self):
        self.regularizer.end()

