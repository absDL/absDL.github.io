import os
import imageio
import sys
from tqdm import tqdm
from skimage import io
from sklearn.model_selection import train_test_split


def replaceLast(s, old, new):  # just a small helper function - replace last iteration of a pattern in a string
    """
    replace last iteration of a pattern in a string
    :param s: str, string
    :param old: str, old ending
    :param new: str, new ending
    :return: str, string with the new ending
    """

    li = s.rsplit(old, 1)
    return new.join(li)


def read_dirs_from_txt_files(txt_file_path):
    """
    Reads a txt file and returns the valid directories
    :param txt_file_path: str, path to textfile
    :return: list, valid dataset directories
    """

    lines = [line.strip('\n') for line in open(txt_file_path)]

    valid_ds = []
    cnt = 0
    for line in lines:
        if len(line):
            if line[0] != '#':
                if os.path.isdir(line):
                    valid_ds.append(line)
                    print('Added dataset: ' + line)
                    cnt += 1

                else:
                    print('Warning - no such directory: ' + line)

    print('A total of %d valid datasets were added' % cnt)

    return valid_ds


def extract_images_from_ds(txt_file_path):
    """
    Lists images from dataset dirs written in txt files
    :param txt_file_path: str, path to txt file
    :return: list, containing the image paths
    """

    ds_list = read_dirs_from_txt_files(txt_file_path)
    if len(ds_list) == 0:
        print('No valid datasets in .txt file: ' + txt_file_path)
        return None

    imList = []
    for ds in ds_list:
        image_list = [os.path.join(ds, im) for im in os.listdir(ds) if im.endswith('.tif')]
        imList += image_list

    if len(imList) == 0:
        print('No images in the datasets listed on: ' + txt_file_path)
        return None

    return imList


def get_train_test_list(inL, centerVer, centerHor):
    """
    Creating the train-validation and the test lists from their respective datasets
    :param inL: int, UNETs input image size [px]
    :param centerVer: int, vertical center of the desired area
    :param centerHor: int, horizontal center of the desired area
    :return:
    inList: list, containing image paths for the training and validation set
    testList: list, containing image paths for the test set
    """

    print('Extracting images WITHOUT atoms (for training and validation)')
    inList = extract_images_from_ds('./woAtoms_ds.txt')
    print('Extracting images WITH atoms (for testing)')
    testList = extract_images_from_ds('./wAtoms_ds.txt')

    image_tiff = io.imread(inList[0])
    if image_tiff.shape[0] > inL:
        cropNsave = True
        if os.path.exists(os.path.dirname(replaceLast(inList[0], '/', '_cropped' + str(inL) + 'px/'))):
            image_tiff = io.imread(replaceLast(inList[0], '/', '_cropped' + str(inL) + 'px/'))
            if image_tiff.shape[0] == inL:  # there is already a cropped version, just correct the paths
                cropNsave = False

        if cropNsave:
            top = int(centerVer - inL / 2)
            left = int(centerHor - inL / 2)

            for i in tqdm(inList + testList):
                image_tiff = io.imread(i)
                image_tiff = image_tiff[top: top + inL, left: left + inL]
                newPath = replaceLast(i, '/', '_cropped' + str(inL) + 'px/')
                if not os.path.exists(os.path.dirname(newPath)):
                    os.mkdir(os.path.dirname(newPath))
                imageio.imwrite(newPath, image_tiff)

        inList = [replaceLast(i, '/', '_cropped' + str(inL) + 'px/') for i in inList]
        testList = [replaceLast(i, '/', '_cropped' + str(inL) + 'px/') for i in testList]

    elif image_tiff.shape[0] < inL:
        print('input tiff size too small!')

    return inList, testList


def save_lists(trainList, valList, testList):
    """
    Saves datasets as lists in order to save the preprocessing time after the script's first run
    :param trainList: list, containing image paths for the training set
    :param valList: list, containing image paths for the validation set
    :param testList: list, containing image paths for the test set
    :return:
    """

    print('number of samples without Atoms= ' + str(len(trainList) + len(valList)))
    print('number of samples with Atoms= ' + str(len(testList)))
    with open("training.lst", "w") as f:
        for s in trainList:
            f.write(str(s) + "\n")
    with open("validation.lst", "w") as f:
        for s in valList:
            f.write(str(s) + "\n")
    with open("testWatoms.lst", "w") as f:
        for s in testList:
            f.write(str(s) + "\n")

    return


def prepare_datasets(args, val_ratio):
    """
    Prepares the datasets for training the UNET architecture. If there are saved datasets they will be used,
    if not they will be generated from the txt files containing the dir paths
    :param args: args, script arguments as described in the parser function
    :param val_ratio: float, validation ratio out of the train-val set
    :return:
    trainList: list, containing image paths for the training set
    valList: list, containing image paths for the validation set
    testList: list, containing image paths for the test set
    """

    if os.path.exists('./training.lst') and os.path.exists('./validation.lst') and os.path.exists('./testWatoms.lst'):
        # use predefined split
        print('Taking data from pre-saved lists')
        trainList = [line.strip() for line in open("training.lst", 'r')]
        valList = [line.strip() for line in open("validation.lst", 'r')]
        testList = [line.strip() for line in open("testWatoms.lst", 'r')]
        print('Training set containing %d samples and validation set containing %d samples.' %
              (len(trainList), len(valList)))
        print('Test set containing %d samples.' % (len(testList)))
        return trainList, valList, testList

    elif os.path.exists('./woAtoms_ds.txt') and os.path.exists('./wAtoms_ds.txt'):
        # make datasets from lists
        print('Taking data from dataset text files')
        inList, testList = get_train_test_list(args.inL, args.centerVer, args.centerHor)
        print('\nCreating training and validation sets, ratio = ' + str(val_ratio))
        trainList, valList = train_test_split(inList, test_size=int(len(inList) * val_ratio))
        print('\nSaving lists')
        save_lists(trainList, valList, testList)
        print('Training set containing %d samples and validation set containing %d samples' %
              (len(trainList), len(valList)))
        return trainList, valList, testList

    else:
        print('ERROR! no data was obtained')
        print('Please make sure you have either .txt files with dataset paths or .lst files with image paths')
        trainList = None
        valList = None
        testList = None

    return trainList, valList, testList


# if __name__ == '__main__':
    # test options
    # read_dirs_from_txt_files('wAtoms_ds.txt')
    # prepare_datasets(476, 442, 804, 0.2)  # why these numbers are hardcoded here?? for temp check
