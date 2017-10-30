from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

class CNN:
    model = None

    def __init__(self, lr=0.001, decay=0):
        opt=Adam(lr=lr, decay=decay)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def Conv2D_bn(self, net, nb_filters, filter_size, strides=(1, 1), padding='same'):
        net = Conv2D(nb_filters, (filter_size, filter_size), strides=strides, padding=padding)(net)
        net = BatchNormalization()(net)
        return Activation('relu')(net)

class DFA(CNN):
    predictions = None
    landmarks = None
    pseudo_labels = None

    def __init__(self, outputs, input_shape, model_type='full', lr=0.001, decay=0, dropout=0):
        image_input = Input(shape=input_shape)
        vgg = self.vgg_block(image_input, 2, 64)
        vgg = self.vgg_block(vgg, 2, 128)
        vgg = self.vgg_block(vgg, 3, 256)
        vgg = self.vgg_block(vgg, 3, 512)
        vgg = self.vgg_block(vgg, 3, 512)
        vgg = Flatten()(vgg)
        vgg = self.Dense_bn(vgg, 4096)
        vgg = Dropout(dropout)(vgg)
        vgg = self.Dense_bn(vgg, 4069)
        vgg = Dropout(dropout)(vgg)
        self.predictions = Dense(outputs, activation='softmax')(vgg)

        self.landmarks = Dense(16)(self.predictions)
        visibility = self.vis_block(self.predictions, model_type=model_type)
        layer = [self.landmarks] + visibility
        self.pseudo_labels = Concatenate([layer])

        self.model = Model(inputs=image_input, outputs=[self.predictions, self.landmarks])
        super().__init__(lr, decay)

    def vgg_block(self, net, nb_conv, nb_filters, filter_size=3):
        for i in range (nb_conv):
            net = self.Conv2D_bn(net, nb_filters, filter_size)
        return MaxPooling2D((2,2), strides=(2,2))(net)

    def Dense_bn(self, net, units):
        net = Dense(units)(net)
        net = BatchNormalization()(net)
        return Activation('relu')(net)

    def vis_block(self, net, model_type='full'):
        visibility = []
        units = None

        if model_type == 'upper':
            units = 6    
        elif model_type == 'lower':
            units = 4
        else:
            units = 8

        for i in range(units):
            visibility.append(Dense(3)(net))

        return visibility

if __name__ == '__main__':
    full = DFA(24, (224,224,1), model_type='full')
    upper = DFA(24, (224,224,1), model_type='upper')
    lower = DFA(24, (224,224,1), model_type='lower')
    cnns = [full, upper, lower]

    for cnn in cnns:
        print(type(cnn).__name__)
        print(cnn.model.summary())
