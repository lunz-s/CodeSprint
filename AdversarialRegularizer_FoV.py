from AdversarialRegularizer import AdversarialRegulariser
import slicing_methods as sl
import numpy as np
import random

class AdversarialSplitting(object):

    def __init__(self, path, NETWORK, BATCH_SIZE, s=0.0, cutoff=20.0, gamma=1.0, lmb=10.0):
        self.m = NETWORK.m
        self.k = NETWORK.k
        self.batch_size = BATCH_SIZE
        IMAGE_SIZE=(None, None, None, None, 1)

        self.regularizer = AdversarialRegulariser(path, IMAGE_SIZE, NETWORK, s=s, cutoff=cutoff, gamma=gamma, lmb=lmb)

    def reset_batch_size(self, size):
        self.batch_size = size


    def train(self, groundTruth, adversarial, learning_rate):
        ground_slices = sl.slice_up(groundTruth, self.m, self.k)
        adv_slices = sl.slice_up(adversarial, self.m, self.k)
        size = ground_slices[(0,0,0)].shape

        ulfs = list(ground_slices.keys())
        length = len(ulfs)
        k = 0
        remaining = self.batch_size + 1
        while remaining > self.batch_size:
            remaining = length - k
            local_bs = min(self.batch_size, remaining)
            bs = (local_bs,) + size
            gt_batch = np.zeros(shape=bs)
            adv_batch = np.zeros(shape=bs)
            for i in range(local_bs):
                gt_batch[i, ...] = ground_slices[ulfs[k]]
                adv_batch[i, ...] = adv_slices[ulfs[k]]
                k += 1
            print('Training on batch of size ' + str(bs))
            self.regularizer.train(gt_batch, adv_batch, learning_rate)

    def test(self, groundTruth, adversarial, random_slice=False):
        ground_slices = sl.slice_up(groundTruth, self.m, self.k)
        adv_slices = sl.slice_up(adversarial, self.m, self.k)
        ulfs = list(ground_slices.keys())

        size = ground_slices[(0, 0, 0)].shape
        bs = (self.batch_size,) + size
        gt_batch = np.zeros(shape=bs)
        adv_batch = np.zeros(shape=bs)

        for k in range(self.batch_size):
            if random_slice:
                ulf = random.choice(ulfs)
            else:
                ulf=ulfs[k]
            gt_batch[k,...]=ground_slices[ulf]
            adv_batch[k,...]=adv_slices[ulf]
        
        print('Evaluation on batch of size ' + str(bs))
        self.regularizer.test(gt_batch, adv_batch)

    def evaluate(self, data):
        full_size=data.shape

        ground_slices = sl.slice_up(data, self.m, self.k)
        size = ground_slices[(0, 0, 0)].shape

        ulfs = list(ground_slices.keys())
        length = len(ulfs)
        k = 0
        remaining = self.batch_size + 1
        grads = {}
        while remaining > self.batch_size:
            remaining = length - k
            local_bs = min(self.batch_size, remaining)
            bs = (local_bs,) + size
            batch = np.zeros(shape=bs)
            for i in range(local_bs):
                batch[i, ...] = ground_slices[ulfs[k + i]]

            gradient_batch = self.regularizer.evaluate(batch)

            for i in range(local_bs):
                grads[ulfs[k + i]] = gradient_batch[i, ...]
            k += local_bs

        val_grads = sl.valid_gradient(grads, self.m, self.k)
        return sl.build_up(val_grads, self.m, full_size)

    def save(self):
        self.regularizer.save()

    def load(self):
        self.regularizer.load()

    def end(self):
        self.regularizer.end()

