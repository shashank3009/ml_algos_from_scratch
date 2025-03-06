import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Embedding, \
    Dense, BatchNormalization, Add, Activation, Reshape, LayerNormalization, MultiHeadAttention, \
        Flatten, MaxPooling2D
from tensorflow.keras.ops.image import extract_patches
from random import sample

import os
import sys
from random import shuffle
sys.path.append('/home/shashank/tapestry/models')

class Encoder(Layer):
    def __init__(self, D, H):
        super(Encoder, self).__init__(name = 'encoder')
        self.norm1 = LayerNormalization()
        self.attention = MultiHeadAttention(num_heads=H, key_dim=D)
        
        self.norm2 = LayerNormalization()
        
        self.dense1 = Dense(2048, activation="relu")
        self.dense2 = Dense(D, activation="relu")
        
    def call(self, encoder_input, input):
        x = self.norm1(encoder_input)
        x = self.attention(x, x, x, attention_mask=input["attention_mask"])
        x_norm = Add()([encoder_input, x])
        x = self.norm2(x_norm)
        x = self.dense1(x)
        x = self.dense2(x)
        x = Add()([x, x_norm])
        
        return x   
        
class PatchNPack(tf.keras.Layer):
    def __init__(self, N, D):
        super(PatchNPack, self).__init__(name = 'patch_n_pack')
        self.pos_x = tf.keras.layers.Embedding(N, D)
        self.pos_y = tf.keras.layers.Embedding(N, D)
        self.projection = Dense(D, activation="relu")
        
    def call(self, input):
        x = self.projection(input["sequence"]) + self.pos_x(input["x_emb_seq"]) + self.pos_y(input["y_emb_seq"])
        return x
        
class Pooling(tf.keras.Layer):
    def __init__(self, num_classes):
        super(Pooling, self).__init__(name = 'masked_pooling')
        self.dense_1 = Dense(128, activation="relu")
        self.softmax = Dense(num_classes, activation="softmax")
    def call(self, encoder_output, input):
        x = tf.matmul(tf.cast(input["pooling_mask"], tf.float32), encoder_output)
        x = self.dense_1(x)
        x = self.softmax(x)
        
        return x

class NaViT(Model):
    def __init__(self, D, n_layers, num_classes):
        super(NaViT, self).__init__(name = 'vision_transformer')

        self.patch_encoder = PatchNPack(N, D)       
        self.encoders = [Encoder(D, H=8) for _ in range(n_layers)]
        self.decoder = Pooling(num_classes)
        
    def call(self, input):
        
        x = self.patch_encoder(input)
        for i in range(len(self.encoders)):
            x = self.encoders[i](x, input)        
        x = self.decoder(x, input)
        
        return x

def nearest_multiple(x, multiple=32):
    x = int(x)
    remainder = x % multiple
    if remainder >= multiple/2:
        x = int(x-remainder+multiple)
    else:
        x = int(x-remainder)
        
    return max(multiple, x)

def resolution_sampling(image, sampling_disb=[64, 128, 196, 256, 320, 384]):
    
    target_resolution = sample(sampling_disb, 1)[0]
    ar = image.height/image.width
    
    new_height = nearest_multiple(ar*target_resolution)
    new_width = nearest_multiple(target_resolution)

    
    target_size = (new_width, new_height) # (height,width)
    
    image = image.resize(target_size)
    
    return image  
    
def token_sampling(height, width, ratio):
    
    ar = height/width           
    sampler = np.ones((height, width), dtype=np.int32)
    indices = np.random.choice(sampler.shape[1]*sampler.shape[0], size=int(sampler.shape[1]*sampler.shape[0]*ratio))
    sampler[np.unravel_index(indices, sampler.shape)] = 0 
    s = (sampler == 1).any(axis=1).sum() / (sampler == 1).any(axis=0).sum() 
    if s == ar:
        return sampler
    else:
        return np.ones((height, width), dtype=np.int32)    
    
def image_to_tokens(image):
    img_array = tf.keras.utils.img_to_array(image).astype(int)
    patches = extract_patches(img_array.astype("float32"), 32).numpy().astype(int)
    
    h = patches.shape[0]
    w = patches.shape[1]
    
    pos_x = np.repeat(np.expand_dims(np.arange(1,w+1), axis=0), repeats=h, axis=0)    
    pos_y = np.repeat(np.expand_dims(np.arange(1,h+1), axis=-1), repeats=w, axis=-1)    
    
    sampler = token_sampling(patches.shape[0], patches.shape[1], 0.2).reshape(patches.shape[0]*patches.shape[1],)
    patches = patches.reshape(patches.shape[0] * patches.shape[1], -1)
    pos_x = pos_x.reshape(pos_x.shape[0] * pos_x.shape[1], )
    pos_y = pos_y.reshape(pos_y.shape[0] * pos_y.shape[1], ) 
    
    dropped_tokens = np.delete(patches, np.where(sampler==0), axis=0)
    pos_x= np.delete(pos_x, np.where(sampler==0), axis=0)
    pos_y= np.delete(pos_y, np.where(sampler==0), axis=0)
    
    assert dropped_tokens.shape[0] == pos_x.shape[0] 
    
    return dropped_tokens, pos_x, pos_y

def compute_pooling_mask(N, N_images , token_list):
    
    mask = np.zeros((N_images, N))
    sum=0
    for i in range(len(token_list)):
        mask[i, sum:sum+token_list[i]] = 1
        sum += token_list[i]
    return mask
        
def compute_attention_mask(N, token_list):
    mask = np.zeros((N,N))
    sum=0
    for i in range(len(token_list)):
        if i == 0:
            mask[:token_list[i],:token_list[i]] = 1
            sum += token_list[i]
        else:
            mask[sum:sum+token_list[i], sum:sum+token_list[i]] = 1
            sum += token_list[i]
            
    return mask

def compute_labels(N_images, num_classes, label_list):
    
    target = np.zeros((N_images, num_classes))
    
    for i, lab in enumerate(label_list):
        target[i, lab] = 1
        
    return target

if __name__=="__main__":

    categories = {'Francis of Assisi':0,
                    'Jerome':1,
                    'John the Baptist':2,
                    'Peter':3,
                    'Virgin Mary':4
                    }

    base_dir = "/home/shashank/tensorflow_datasets/downloads/ArtDL_small_2"

    all_files = []

    for category in categories.keys():
        for file in os.listdir(os.path.join(base_dir, category)):
            all_files.append((os.path.join(base_dir, category, file), categories[category]))
            
    shuffle(all_files)

    N = 512
    N_images = 8
    num_classes=5
    batch_size = 4

    def generator():
        i=0
        image_count = 0
        seq_len = 0
        flag = True  
        token_list = []
        labels_list = []
        for (image_path, label) in all_files:
            image = tf.keras.utils.load_img(image_path)
            image = resolution_sampling(image)
            tokens, pos_x, pos_y = image_to_tokens(image)    
            
            if tokens.shape[0] > 510:
                continue    
            
            
            if flag:
                token_list.append(tokens.shape[0])
                labels_list.append(label)
                sequence = tokens
                x_emb_seq = pos_x
                y_emb_seq = pos_y
                image_count +=1
                flag = False        
            
            else:
                if tokens.shape[0] <= N - sequence.shape[0]:
                    token_list.append(tokens.shape[0])
                    labels_list.append(label)
                    sequence = np.concatenate((sequence, tokens), axis=0)
                    x_emb_seq = np.concatenate((x_emb_seq, pos_x), axis=0)
                    y_emb_seq = np.concatenate((y_emb_seq, pos_y), axis=0)
                    image_count +=1
                else:
                    
                    labels = compute_labels(N_images, num_classes, labels_list)
                    attention_mask = compute_attention_mask(N, token_list)
                    pooling_mask = compute_pooling_mask(N, N_images , token_list)
                
                        
                    if N-sequence.shape[0] > 0:
                        sequence = np.concatenate((sequence, np.zeros((N-sequence.shape[0], 3072))), axis=0)
                        x_emb_seq = np.concatenate((x_emb_seq, np.zeros(N - x_emb_seq.shape[0])), axis=0)
                        y_emb_seq = np.concatenate((y_emb_seq, np.zeros(N - y_emb_seq.shape[0])), axis=0)
                                    
                    token_list = []
                    labels_list = []
                    image_count=0
                    yield {"sequence": sequence, 
                        "x_emb_seq": x_emb_seq,
                        "y_emb_seq": y_emb_seq,
                        "attention_mask": attention_mask,
                        "pooling_mask": pooling_mask
                        }, labels    
                    token_list.append(tokens.shape[0])
                    labels_list.append(label)   
                    sequence = tokens
                    x_emb_seq = pos_x
                    y_emb_seq = pos_y
                    image_count +=1        
                    
                    
            if image_count == N_images:            
                
                image_count = 0 
                
                labels = compute_labels(N_images, num_classes, labels_list)
                attention_mask = compute_attention_mask(N, token_list)
                pooling_mask = compute_pooling_mask(N, N_images , token_list)
                
                if N-sequence.shape[0] > 0:
                    sequence = np.concatenate((sequence, np.zeros((N-sequence.shape[0], 3072))), axis=0)
                    x_emb_seq = np.concatenate((x_emb_seq, np.zeros(N - x_emb_seq.shape[0])), axis=0)
                    y_emb_seq = np.concatenate((y_emb_seq, np.zeros(N - y_emb_seq.shape[0])), axis=0)
                
                token_list = []
                labels_list = []  
                        
                yield {"sequence": sequence, 
                    "x_emb_seq": x_emb_seq,
                    "y_emb_seq": y_emb_seq,
                    "attention_mask": attention_mask,
                    "pooling_mask": pooling_mask
                    }, labels
                token_list.append(tokens.shape[0])
                labels_list.append(label)                 
                sequence = tokens
                x_emb_seq = pos_x
                y_emb_seq = pos_y
                image_count +=1 
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "sequence": tf.TensorSpec(shape=(N,3072), dtype=tf.int32),
                "x_emb_seq": tf.TensorSpec(shape=(N,), dtype=tf.int32),
                "y_emb_seq": tf.TensorSpec(shape=(N,), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(N,N), dtype=tf.int32),
                "pooling_mask": tf.TensorSpec(shape=(N_images, N), dtype=tf.int32)
            }, 
            tf.TensorSpec(shape=(N_images, num_classes), dtype=tf.int32)
        )
    )

    train, val = tf.keras.utils.split_dataset(dataset, left_size=0.8, right_size=0.2) # takes a few minutes to complete

    train_dataset = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tf.keras.backend.clear_session()
        
        
    D = 768
    n_layers = 6
    navit_model = NaViT(D=D, n_layers=n_layers, num_classes=num_classes)

    loss = tf.keras.losses.CategoricalFocalCrossentropy()

    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(name="recall")]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    checkpoint_filepath = '/home/shashank/cv/navit_model_training/checkpoints/{epoch:02d}-{val_recall:.2f}.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_recall',
        mode='max',
        save_best_only=True)

    log_dir = '/home/shashank/cv/navit_model_training/tensorboard/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    csv_logger = tf.keras.callbacks.CSVLogger('/home/shashank/cv/navit_model_training/training.log')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_recall', 
                                                    min_delta=0.05, patience=5, mode="max")

    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        elif epoch < 10:
            return lr * 0.5
        
        elif epoch < 20:
            return lr * 0.1
        else:
            return lr * 0.01

    lr_schedular = tf.keras.callbacks.LearningRateScheduler(scheduler)

    navit_model.compile(loss=loss, metrics=metrics, optimizer=optimizer) 

    navit_model.fit(train_dataset, validation_data=val_dataset, epochs=25, callbacks=[model_checkpoint_callback, 
                                                                        tensorboard_callback, 
                                                                        csv_logger, early_stopping, lr_schedular])
    