import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, \
    Dense, BatchNormalization, Add, Activation
import sys

sys.path.append('/home/shashank/tapestry/models')

class ConvolutionLayer(Layer):
    def __init__(self, filters, stride=1, kernel_size=3, padding="valid"):
        super().__init__()
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, stride=stride, padding="same")
        self.bn_layer = BatchNormalization()
        self.activation = Activation(activation="relu")
        
    def call(self, input, training=True):
        x = self.conv(input)
        x = self.bn_layer(x, training=training)
        x = self.activation(x)
        
        return x
    
class ResidualBlock(Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        
        self.conv1 = ConvolutionLayer(filters=filters, stride=stride)
        self.conv2 = ConvolutionLayer(filters=filters, stride=1)  
        self.activation = Activation("relu")

        
        self.dotted = stride!=1
        
        if self.dotted:
            self.conv_dotted = ConvolutionLayer(filters=filters, kernel_size=1, stride=2)
        
    def call(self, input, training=True):
        
        x = self.conv1(input, training=training)        
        x = self.conv2(x, training=training)
        
        if self.dotted:
            input = self.conv_dotted(input, training=training)
            x = Add()[x, input]
        else:
            x = Add()[x, input]
        
        return self.activation(x)
    
num_classes = 10
class Resnet34(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1_1 = ConvolutionLayer(filters=64, kernel_size=7, stride=2, padding="same")
        self.max_pool1 = MaxPool2D(pool_size=(3,3), strides=2)
        
        self.conv2_1 = ResidualBlock(64)
        self.conv2_2 = ResidualBlock(64)
        self.conv2_3 = ResidualBlock(64)
        
        self.conv3_1 = ResidualBlock(128, 2)
        self.conv3_2 = ResidualBlock(128)
        self.conv3_3 = ResidualBlock(128)
        self.conv3_4 = ResidualBlock(128)
        
        self.conv4_1 = ResidualBlock(256, 2)
        self.conv4_2 = ResidualBlock(256)
        self.conv4_3 = ResidualBlock(256)
        self.conv4_4 = ResidualBlock(256)
        self.conv4_5 = ResidualBlock(256)
        self.conv4_6 = ResidualBlock(256)
        
        self.conv5_1 = ResidualBlock(512, 2)
        self.conv5_2 = ResidualBlock(512)
        self.conv5_3 = ResidualBlock(512)
        
        self.avg_pool = GlobalAveragePooling2D()
        self.fc_1000 = Dense(1000, activation="relu")
        self.fc_softmax = Dense(num_classes, activation="softmax")
        
    def call(self, input, training=True):
        
        x = self.conv1_1(input, training=training)
        x = self.max_pool1(x)
        
        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.conv2_3(x, training=training)
        
        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.conv3_4(x, training=training)
        
        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        x = self.conv4_3(x, training=training)
        x = self.conv4_4(x, training=training)
        x = self.conv4_5(x, training=training)
        x = self.conv4_6(x, training=training)
        
        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        x = self.conv5_3(x, training=training)
        
        x = self.avg_pool(x)
        x = self.fc_1000(x)
        x = self.fc_softmax(x)
               
        return x      