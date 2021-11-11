import os
import random
import torch
import numpy
import torch.backends.cudnn
from src.data.prepare_data import Preparedata
from src.model.CNN import CNN
from src.model.run import Run
from src.parameters.parameters import Parameters


class Controller(Parameters):

    def __init__(self):

        # prepare the data
        self.data = Preparedata(Parameters)
        self.data.prepare_data(Parameters)

    def initialise_model(self):

        # prepare the model
        self.model = CNN(Parameters)

    def train(self):

        # train the model and return train and dev scores
        self.ran_model = Run(self.model, self.data, Parameters)
        macro_fscores_train, macro_fscores_dev = self.ran_model.train()

        return macro_fscores_train, macro_fscores_dev

    def test(self):

        # run the model on the test data
        macro_fscore_test = self.ran_model.test()
        return macro_fscore_test

def block_seeds(seed):

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(int(seed))
    random.seed(int(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':

    params = Parameters()

    print(f'--------------------- Preparing data ---------------------')
    controller = Controller()

    train, dev, test = [], [], []
    for seed in params.seeds:
        print(f'--------------- Seed: {seed} ---------------')
        block_seeds(seed)
        controller.initialise_model()

        macro_fscores_train, macro_fscores_dev = controller.train()
        macro_fscore_test = controller.test()

        train.append(macro_fscores_train)
        dev.append(macro_fscores_dev)
        test.append(macro_fscore_test)

    mean_train = numpy.matrix(train).mean(axis=0).tolist()[0]
    mean_dev = numpy.matrix(dev).mean(axis=0).tolist()[0]

    print(f'Macro f-score on the test set: {round(numpy.mean(test), 2)}')
