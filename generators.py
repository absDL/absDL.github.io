import os
import numpy as np
from skimage import io
from tqdm import tqdm


def npGenerator(dirs, ord, inL, outL, mask):
    """
    Generates X Y pairs of batches of the masked images and their correspondent GT
    :param dirs: list, containing image paths of the dataset
    :param ord: list, indices of the desired images
    :param inL: int, UNETs input image size [px]
    :param outL: int, UNETs output size [px]
    :param mask: binary 2D np.array, mask of the OD image with zeros on the target area
    :return:
    X: np.array, the masked input image
    Y: np.array, the target image (or the unmasked input image)
    """

    X = np.empty([len(ord), inL, inL])
    Y = np.empty([len(ord), outL, outL])
    for i in range(len(ord)):
        image_tiff = np.array(io.imread(dirs[ord[i]]))
        OD_image = np.array(image_tiff, dtype=np.dtype('float32')) / 4294967295
        Y[i, :, :] = OD_image[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                     int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]
        x_temp = OD_image * mask
        X[i, :, :] = x_temp + (1 - mask) * 0.5

    return X, Y


def initialize_model(model):
    """
    Initializes the model, either from scratch or from previous checkpoint
    :param model: the UNET model
    :return:
    model: model, the UNET model with weights loaded from the latest checkpoint
    epochNum: int, epoch number from the latest checkpoint
    trainLoss: array or array-like object, training loss from the latest checkpoint
    valLoss: array or array-like object, validation loss from the latest checkpoint
    """

    if not os.path.isdir('models'):
        os.mkdir('models')

    # list the model files
    modelsList = [os.path.join('./models/', each) for each in os.listdir('./models/') if each.endswith('.h5')]
    if not len(modelsList):
        print('starting from scratch')
        epochNum = 1
        trainLoss = []
        valLoss = []
    else:
        modelFile = modelsList[-1]
        epochNum = int(modelFile.split('/')[-1].split('_')[-1].split('.')[0])
        print('starting from the end of epoch ' + str(epochNum))
        model.load_weights(modelFile)
        trainLoss = np.load('training_loss_history.npy')
        valLoss = np.load('validation_history.npy')

    return model, epochNum, trainLoss, valLoss


def generate_reference_loss(inL, outL, trainList, valList):
    """
    Generates the reference loss for the experiment
    :param inL: int, UNETs input image size [px]
    :param outL: int, UNETs output size [px]
    :param trainList: list, containing image paths for the training set
    :param valList: list, containing image paths for the validation set
    :return: np.array, containing the reference MSE for the experiment
    """

    if os.path.exists(os.getcwd() + '/referenceMSEtrainA.npy'):
        trainAloss = np.load('referenceMSEtrainA.npy')
        trainRloss = np.load('referenceMSEtrainR.npy')
        valAloss = np.load('referenceMSEvalA.npy')
        valRloss = np.load('referenceMSEvalR.npy')

    else:
        trainA = [s for s in trainList if "A_no_atoms" in s]
        trainAloss = np.empty(len(trainA))
        for i in tqdm(range(len(trainA))):
            refDir = trainA[i].replace("A_no_atoms", "R_no_atoms")
            Y = np.array(np.array(io.imread(trainA[i])[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                  int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]), dtype=np.dtype('float32')) \
                / 4294967295
            ref = np.array(np.array(io.imread(refDir)[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                    int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]), dtype=np.dtype('float32')) \
                / 4294967295
            trainAloss[i] = np.average((ref - Y) ** 2)
        np.save('referenceMSEtrainA.npy', trainAloss)

        trainR = [s for s in trainList if "R_no_atoms" in s]
        trainRloss = np.empty(len(trainR))
        for i in tqdm(range(len(trainR))):
            refDir = trainR[i].replace("R_no_atoms", "A_no_atoms")
            Y = np.array(np.array(io.imread(trainR[i])[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                  int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                         dtype=np.dtype('float32')) / 4294967295
            ref = np.array(np.array(io.imread(refDir)[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                    int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                           dtype=np.dtype('float32')) / 4294967295
            trainRloss[i] = np.average((ref - Y) ** 2)
        np.save('referenceMSEtrainR.npy', trainRloss)

        valA = [s for s in valList if "A_no_atoms" in s]
        valAloss = np.empty(len(valA))
        for i in tqdm(range(len(valA))):
            refDir = valA[i].replace("A_no_atoms", "R_no_atoms")
            Y = np.array(np.array(io.imread(valA[i])[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                  int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                         dtype=np.dtype('float32')) / 4294967295
            ref = np.array(np.array(io.imread(refDir)[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                    int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                           dtype=np.dtype('float32')) / 4294967295
            valAloss[i] = np.average((ref - Y) ** 2)
        np.save('referenceMSEvalA.npy', valAloss)

        valR = [s for s in valList if "R_no_atoms" in s]
        valRloss = np.empty(len(valR))
        for i in tqdm(range(len(valR))):
            refDir = valR[i].replace("R_no_atoms", "A_no_atoms")
            Y = np.array(np.array(io.imread(valR[i])[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                  int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                         dtype=np.dtype('float32')) / 4294967295
            ref = np.array(np.array(io.imread(refDir)[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                                    int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]),
                           dtype=np.dtype('float32')) / 4294967295
            valRloss[i] = np.average((ref - Y) ** 2)
        np.save('referenceMSEvalR.npy', valRloss)

    referenceMSE = np.mean(np.concatenate((trainAloss, trainRloss, valAloss, valRloss)))
    print('reference MSE is ' + str(referenceMSE))

    return referenceMSE


if __name__ == '__main__':
    from unet import unet_model

    model = unet_model(476, 192)
    model = initialize_model(model)

    from preprocessing import prepare_datasets

    # trainList, valList, testList = prepare_datasets(476, 442, 804, 0.2)
    trainList, valList, testList = prepare_datasets(804, 0.2)
    referenceMSE = generate_reference_loss(476, 192, trainList, valList)
