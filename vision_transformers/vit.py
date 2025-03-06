import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Embedding, \
    Dense, BatchNormalization, Add, Activation, Reshape, LayerNormalization, MultiHeadAttention, \
        Flatten, MaxPooling2D
from tensorflow.keras.ops.image import extract_patches

import sys
sys.path.append('/home/shashank/tapestry/models')
import tensorflow_models as tfm

class ClsEmbedding(Layer):
    def __init__(self):
        super(ClsEmbedding, self).__init__(name = 'cls_embedding')
        self.cls_embedding = self.add_weight(name="cls_learnable_embedding", 
                                             shape=(1,1,D), initializer="glorot_uniform", trainable=True)
           
    def call(self, input):    
        # cls_token = tf.expand_dims(self.cls_embedding, axis=0)
        cls_token = tf.tile(self.cls_embedding, [tf.shape(input)[0], 1, 1]) 
        x = tf.concat([cls_token, input], axis=1)      
        
        return x
    

class PatchEncoder(Layer):
    def __init__(self, N, P, D, C=3):
        super(PatchEncoder, self).__init__(name = 'patch_encoder')
        
        self.flatten = Reshape((N, (P**2)*C)) # Target shape. Tuple of integers, does not include the samples dimension (batch size).
        self.projection = Dense(D, activation="relu")
    
    def call(self, input):
        x = extract_patches(input, P)
        x = self.flatten(x)
        x = self.projection(x)
        
        return x
    
class MLPHead(Layer):
    def __init__(self, D, num_classes):
        super(MLPHead, self).__init__(name = 'mlp_head')       
        
        self.dense_1 = Dense(D, activation="relu") 
        self.dense_2 = Dense(num_classes, activation="softmax")
    
    def call(self, x):
        cls_head = x[:,0,:]        
        output = self.dense_1(cls_head)
        output = self.dense_2(output)
        
        return output
    
class Encoder(Layer):
    def __init__(self, D, H):
        super(Encoder, self).__init__(name = 'encoder')
        self.norm1 = LayerNormalization()
        self.attention = MultiHeadAttention(num_heads=H, key_dim=D)
        
        self.norm2 = LayerNormalization()
        
        self.dense1 = Dense(2048, activation="relu")
        self.dense2 = Dense(D, activation="relu")
        
    def call(self, input):
        x = self.norm1(input)
        x = self.attention(x, x)
        x_norm = Add()([input, x])
        x = self.norm2(x_norm)
        x = self.dense1(x)
        x = self.dense2(x)
        x = Add()([x, x_norm])
        
        return x   
        
class ViT(Model):
    def __init__(self, batch_size, hidden_dim, n_patches, patch_size, n_layers, num_classes):
        super(ViT, self).__init__(name = 'vision_transformer')

        self.patch_encoder = PatchEncoder(n_patches, patch_size, hidden_dim)
        self.cls_head = ClsEmbedding()
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(max_length=1+n_patches)        
        
        self.encoders = [Encoder(hidden_dim, H=8) for _ in range(n_layers)]
        self.mlp = MLPHead(hidden_dim, num_classes)
        
    def call(self, input):
        x = self.patch_encoder(input)
        x = self.cls_head(x)
        x = x + self.pos_emb(tf.range(1 + N)[:,None])
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
        
        x = self.mlp(x)
        
        return x 

if __name__=="__main__":
    
    tf.keras.backend.clear_session()
    
    D = 768
    H = 512
    W = 512
    P = 32
    C = 3
    BATCH_SIZE = 4
    N = int(H*W/(P**2))

    n_layers = 6

    num_classes = 5
    
    BATCH_SIZE = 4
    image_dir = "/home/shashank/tensorflow_datasets/downloads/ArtDL_small_2"
    train, test = tf.keras.utils.image_dataset_from_directory(image_dir, labels="inferred", image_size=(512,512), 
                                                    batch_size=BATCH_SIZE, shuffle=True, label_mode="categorical", seed=99,
                                                    validation_split=0.2, subset="both")
    
    vit_model = ViT(batch_size=4, hidden_dim=768, n_patches=N, patch_size=32, n_layers=6, num_classes=5)

    loss = tf.keras.losses.CategoricalFocalCrossentropy()

    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(name="recall")]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    checkpoint_filepath = '/home/shashank/cv/vit_model_training/checkpoints/{epoch:02d}-{val_recall:.2f}.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_recall',
        mode='max',
        save_best_only=True)

    log_dir = '/home/shashank/cv/vit_model_training/tensorboard/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    csv_logger = tf.keras.callbacks.CSVLogger('/home/shashank/cv/vit_model_training/training.log')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_recall', 
                                                    min_delta=0.05, patience=5, mode="max")

    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        elif epoch < 10:
            return lr * 0.5
        
        elif lr < 20:
            return lr * 0.1
        else:
            return lr * 0.01

    lr_schedular = tf.keras.callbacks.LearningRateScheduler(scheduler)

    vit_model.compile(loss=loss, metrics=metrics, optimizer=optimizer) 

    
    dummy_input = tf.keras.Input(shape=(512, 512, 3))
    build_model = vit_model.call(dummy_input)
    
    vit_model.fit(train, validation_data=test, epochs=25, callbacks=[model_checkpoint_callback, 
                                                                     tensorboard_callback, 
                                                                     csv_logger, early_stopping, lr_schedular])
