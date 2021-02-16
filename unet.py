import numpy as np
from keras.backend import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Reshape, ReLU
from keras.layers import Cropping2D,SeparableConv2D
from keras.layers.merge import concatenate
from keras.models import Model

def unet_model(inL, outL):
	'''
	Builds a 4-story UNET model
	:param inL: int, UNETs input image size [px]
	:param outL: int, UNETs output size [px]
	:return: model, the UNET model
	'''
	
    d1 = 64
    d2 = 128
    d3 = 256
    d4 = 512
    d5 = 1024
    b_momentum=0.99

    minS = 22 # smallest size of layer at the bottom of the U
    inS = minS
    for i in range(4):
        inS = (inS+4)*2
    inS = inS+4

    upS = [inS-4]
    for i in range(3):
        s = int(upS[-1]/2-4)
        upS = np.append(upS,s)

    downS = minS
    downS = [int(downS*2)]
    for i in range(3):
        s = int(downS[-1]-4)*2
        downS = np.append(downS,s)
    downS = np.flip(downS)
    lastS = downS[0]-6
    crop = (upS-downS)/2

    cropL = int((lastS-outL)/2)
    cropL1 = int(crop[0])
    cropL2 = int(crop[1])
    cropL3 = int(crop[2])
    cropL4 = int(crop[3])

    inp = Input(shape=(inL,inL))
    inp1 = Reshape(target_shape=(inS,inS,1))(inp)

    conv1 = Conv2D(d1, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(inp1)
    conv1 = BatchNormalization(momentum=b_momentum)(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(d1, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv1)
    conv1 = BatchNormalization(momentum=b_momentum)(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(d2, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(pool1)
    conv2 = BatchNormalization(momentum=b_momentum)(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(d2, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv2)
    conv2 = BatchNormalization(momentum=b_momentum)(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 =  Conv2D(d3, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(pool2)
    conv3 = BatchNormalization(momentum=b_momentum)(conv3)
    conv3 = ReLU()(conv3)
    conv3 =  Conv2D(d3, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv3)
    conv3 = BatchNormalization(momentum=b_momentum)(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(d4, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(pool3)
    conv4 = BatchNormalization(momentum=b_momentum)(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(d4, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv4)
    conv4 = BatchNormalization(momentum=b_momentum)(conv4)
    conv4 = ReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    convJ = SeparableConv2D(d5, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(pool4)
    convJ = BatchNormalization(momentum=b_momentum)(convJ)
    convJ = ReLU()(convJ)
    convJ = SeparableConv2D(d5, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(convJ)
    convJ = BatchNormalization(momentum=b_momentum)(convJ)
    convJ = ReLU()(convJ)

    up5 = Conv2DTranspose(d4, (2, 2), strides=(2, 2), padding='valid')(convJ)
    crop4 = Cropping2D(cropping=(cropL4, cropL4), data_format="channels_last")(conv4)
    merge5 = concatenate([crop4,up5], axis = 3)
    conv5 = Conv2D(d4, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(merge5)
    conv5 = BatchNormalization(momentum=b_momentum)(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(d4, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv5)
    conv5 = BatchNormalization(momentum=b_momentum)(conv5)
    conv5 = ReLU()(conv5)

    up6 = Conv2DTranspose(d3, (2, 2), strides=(2, 2), padding='valid')(conv5)
    crop3 = Cropping2D(cropping=(cropL3, cropL3), data_format="channels_last")(conv3)
    merge6 = concatenate([crop3,up6], axis = 3)
    conv6 = Conv2D(d3, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(merge6)
    conv6 = BatchNormalization(momentum=b_momentum)(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv2D(d3, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv6)
    conv6 = BatchNormalization(momentum=b_momentum)(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv2DTranspose(d2, (2, 2), strides=(2, 2), padding='valid')(conv6)
    crop2 = Cropping2D(cropping=(cropL2, cropL2), data_format="channels_last")(conv2)
    merge7 = concatenate([crop2,up7], axis = 3)
    conv7 = Conv2D(d2, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(merge7)
    conv7 = BatchNormalization(momentum=b_momentum)(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv2D(d2, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv7)
    conv7 = BatchNormalization(momentum=b_momentum)(conv7)
    conv7 = ReLU()(conv7)

    up8 = Conv2DTranspose(d1, (2, 2), strides=(2, 2), padding='valid')(conv7)
    crop1 = Cropping2D(cropping=(cropL1, cropL1), data_format="channels_last")(conv1)
    merge8 = concatenate([crop1,up8], axis = 3)
    conv8 = Conv2D(d1, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(merge8)
    conv8 = BatchNormalization(momentum=b_momentum)(conv8)
    conv8 = ReLU()(conv8)
    conv8 = Conv2D(d1, (3, 3),
                   padding='valid', kernel_initializer='glorot_normal')(conv8)
    conv8 = BatchNormalization(momentum=b_momentum)(conv8)
    conv8 = ReLU()(conv8)

    output = Conv2D(1, (3, 3), padding='valid')(conv8)
    output = Cropping2D(cropping=(cropL, cropL), data_format="channels_last")(output)
    output = Reshape((outL,outL))(output)

    model = Model(inputs=inp, outputs=output)

    return model

if __name__=='__main__':
    # test unet
    model = unet_model(476, 192)
    model.summary()