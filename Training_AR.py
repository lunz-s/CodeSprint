from AdversarialRegularizer import AdversarialRegulariser
from networks import AlexNet_3D

BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 2
STEPS = 5000


# Parameter choices. Heuristic in the BWGAN paper: Choose GAMMA as average dual norm of clean image
# LMB should be bigger than product of norm times dual norm.

# For s=0.0, this implies GAMMA =1.0
# For s=1.0, have GAMMA = 10.0 as realisitc value
S = 0.0
LMB = 10.0
GAMMA = 1.0
CUTOFF = 20.0



saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/Cutoff_20/Translation_Augmentation'
regularizer = AdversarialRegulariser(saves_path, IMAGE_SIZE=(None, 256,256,256,1), NETWORK=AlexNet_3D,
                                     s=S, cutoff=20.0, lmb=LMB, gamma=GAMMA)


def evaluate():
    gt, adv = get_batch(eval_data=True, noise_levels=['01', '016'], methods=['EM', 'SGD'])
    regularizer.test(groundTruth=gt, adversarial=adv)


def train(steps):
    for k in range(steps):
        gt, adv = get_batch(eval_data=False)
        regularizer.train(groundTruth=gt, adversarial=adv, learning_rate=LEARNING_RATE)
        if k%50==0:
            evaluate()
    regularizer.save()


for k in range(LOOPS):
    train(STEPS)

LEARNING_RATE = 0.00002

for k in range(LOOPS):
    train(STEPS)

