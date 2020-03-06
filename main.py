import gc, os
import imageio
from keras import backend as K
from keras.optimizers import Adam, SGD
import argparse
from tqdm import tqdm

from plotter import plot_runtime_error, plot_single_comparison
from generators import *
from preprocessing import prepare_datasets, replaceLast
from unet import unet_model

K.set_image_data_format('channels_last')

def infer_single_img(inL, outL, mask, model, img_path):
    OD_image = np.array(np.array(io.imread(img_path)), dtype=np.dtype('float32')) / 4294967295
    Y = OD_image[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
        int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]
    x_masked = OD_image * mask
    X = x_masked + (1 - mask) * 0.5
    Y_prime = model.predict(X[np.newaxis, :, :])
    Y_prime = Y_prime[0, :, :]

    return X, Y_prime, Y


def save_tif(tif_path, tif_data):
    T = np.array(tif_data*4294967295, dtype=np.dtype('uint32'))
    imageio.imwrite(tif_path, T)
    
    return


def save_bin(bin_path, bin_data):
    output_file = open(bin_path, 'wb')
    bin_data.tofile(output_file)
    output_file.close()
    
    return

def train_model(args, outL, mask, model, trainList, valList, epochNum, referenceMSE, trainLoss, valLoss):
    inL = args.inL
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    trainBatches = len(trainList) // batch_size
    valBatches = len(valList) // batch_size

    while epochNum <= max_epochs:
        shufTrain = np.arange(len(trainList))
        shufVal = np.arange(len(valList))
        np.random.shuffle(shufTrain)
        np.random.shuffle(shufVal)
        curTrainLoss = 0
        curValLoss = 0

        for i in tqdm(range(trainBatches)):
            X, Y = npGenerator(trainList, shufTrain[i * batch_size:(i + 1) * batch_size], inL, outL, mask)
            curTrainLoss = curTrainLoss + model.train_on_batch(x=X, y=Y) / trainBatches
        print('epoch ' + str(epochNum) + ' train loss = ' + str(curTrainLoss))

        for i in tqdm(range(valBatches)):
            X, Y = npGenerator(valList, shufVal[i * batch_size:(i + 1) * batch_size], inL, outL, mask)
            curValLoss = curValLoss + model.test_on_batch(x=X, y=Y) / valBatches
        print('epoch ' + str(epochNum) + ' validation loss = ' + str(curValLoss))

        trainLoss = np.append(trainLoss, curTrainLoss)
        valLoss = np.append(valLoss, curValLoss)

        plot_runtime_error(epochNum, trainLoss, valLoss, referenceMSE)

        modelFile = 'models/epoch_' + str(epochNum).zfill(4) + '.h5'
        model.save(modelFile, include_optimizer=False)
        if args.dont_save_models and (epochNum > 1):
            os.remove('models/epoch_' + str(epochNum - 1).zfill(4) + '.h5') 

        np.save('training_loss_history.npy', trainLoss)
        np.save('validation_history.npy', valLoss)
        gc.collect()

        epochNum = epochNum + 1

    return model

def generate_mask(inL, maskR):
    scale = np.arange(inL)
    mask = np.zeros((inL, inL))
    mask[(scale[np.newaxis, :] - (inL - 1) / 2) ** 2 + (scale[:, np.newaxis] - (inL - 1) / 2) ** 2 > maskR ** 2] = 1

    return mask

def get_parser():
    parser = argparse.ArgumentParser(description='Single-Shot Absorption Imaging of Ultracold Atoms '
                                                 'Using Deep-Neural-Network')
    parser.add_argument('-il', '--inL', default=476, type=int, help='UNETs input image size [px]')
    parser.add_argument('-mr', '--maskR', default=95, type=int, help='Mask radius [px]')
    parser.add_argument('-cv', '--centerVer', default=476, type=int,
                        help='vertical position of the center of the atom cloud [px]')
    parser.add_argument('-ch', '--centerHor', default=804, type=int,
                        help='horizontal position of the center of the atom cloud [px]')
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-lr', '--learning_rate', default=5e-6, type=float)
    parser.add_argument('-sgd', '--SGD', default=False, action='store_true')
    parser.add_argument('-e', '--max_epochs', default=1000, type=int)
    parser.add_argument('-dsm', '--dont_save_models', action='store_true', default=False,
                        help='save all models to files')
    parser.add_argument('-src', '--skip_reference_comparison', action='store_true', default=False,
                        help='avoid comprison to reference (double-shot absorption imaging).'
                             'use this flag if you don\'t have the same dataset structure as the original: '
                             'A_no_atoms, R_no_atoms, A_with_atoms, R_with_atoms')

    return parser

if __name__=='__main__':
    ## params
    parser = get_parser()
    args = parser.parse_args()

    outL = 2 * args.maskR  # Output size
    mask = generate_mask(args.inL, args.maskR)

    ## get data
    trainList, valList, testList = prepare_datasets(args, 0.2)


    ## build model
    K.clear_session()
    model = unet_model(args.inL, outL)
    model.summary()  # display model summary
    model, epochNum, trainLoss, valLoss = initialize_model(model)
    if args.SGD:
        opt = SGD(lr=1e-2, momentum=0.9, decay=1e-4/args.max_epochs)
    else:
        opt = Adam(lr=args.learning_rate)
    model.compile(optimizer=opt, loss='mse')
    model.compile(optimizer=opt, loss='mse')


    ## calculate referance loss
    if not args.skip_reference_comparison:
        referenceMSE = generate_referance_loss(args.inL, outL, trainList, valList)
    else:
        referenceMSE = None


    ## train model
    model = train_model(args, outL, mask, model, trainList, valList, epochNum, referenceMSE, trainLoss, valLoss)

# TODO: move next two sections to a seperate file (with 1. load model 2. load images 3. display), and clear from main

    ## infer and plot
    # on eval image
    random_id = np.random.randint(0, len(valList))
    image_name = valList[random_id]
    print(image_name)  # use eiter these three lines or set image_name= ""...
    X, pred, Y = infer_single_img(args.inL, outL, mask, model, image_name)
    # toggle to 'True' to save tifs
    if False:
        save_tif('X.tif', X)
        save_tif('pred.tif', pred)
        save_tif('Y', Y)
    plot_single_comparison(X, pred, Y, pred)

    # on test image
    if referance_loss:
        random_id = np.random.randint(0, len(testList))
        image_name = testList[random_id]
        print(image_name)  # use eiter these three lines or set image_name= ""...
        X, pred, Y = infer_single_img(args.inL, outL, mask, model, image_name)
        ref_image = np.array(np.array(io.imread(image_name.replace('A_with_atoms', 'R_with_atoms'))),
                             dtype=np.dtype('float32')) / 4294967295
        ref_image = ref_image[int(inL / 2 - outL / 2):int(inL / 2 + outL / 2),
                    int(inL / 2 - outL / 2):int(inL / 2 + outL / 2)]
        plot_single_comparison(X, pred, Y, ref_image)

# TODO: move next two sections to a seperate file (with 1. load model 2. load images 3. store predictions), and keep in main

    # on all test images + save
    newPath = os.path.dirname(testList[0]) + '_predicted' + str(epochNum) + '/'
    if not os.path.isdir(newPath):
        os.mkdir (newPath)
    for inpPath in tqdm(testList):
        X, Y_prime, Y = infer_single_img(args.inL, outL, mask, model, inpPath)
        binPath = replaceLast(inpPath, '/', '_predicted' + str(epochNum) + '/').replace('.tif', '.bin')
        save_bin(binPath, Y_prime)


    # on all validation images + save
    for inpPath in tqdm(valList):
        X, Y_prime, Y = infer_single_img(args.inL, outL, mask, model, inpPath)
        
        binPath = replaceLast(inpPath, '/', '_validation_cropped/').replace('.tif', '.bin')
        if not os.path.isdir(os.path.dirname(binPath)):
            os.mkdir (os.path.dirname(binPath))
        if not os.path.isfile(binPath):
            save_bin(binPath, Y)
        
        binPath = replaceLast(inpPath, '/', '_predicted' + str(epochNum) + '/').replace('.tif', '.bin')
        if not os.path.isdir(os.path.dirname(binPath)):
            os.mkdir (os.path.dirname(binPath))
        save_bin(binPath, Y_prime)
