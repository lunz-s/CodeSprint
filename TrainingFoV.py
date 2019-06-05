from AdversarialRegularizer_FoV import AdversarialSplitting
from networks_FoV import ResNetL2

BATCH_SIZE = 1
LEARNING_RATE = 0.00005
LOOPS = 2
STEPS = 5000


# Parameter choices. Heuristic in the BWGAN paper: Choose GAMMA as average dual norm of clean image
# LMB should be bigger than product of norm times dual norm.


saves_path = '/local/scratch/public/sl767/SPA/Saves/Adversarial_Regulariser/Cutoff_20/Translation_Augmentation'
regularizer = AdversarialSplitting(saves_path, NETWORK=ResNetL2, BATCH_SIZE=4)


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