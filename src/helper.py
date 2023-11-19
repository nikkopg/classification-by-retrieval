import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from customLayer import AggregationLayer
import tensorflow as tf

img_size = 224 # default size

def crop_image(image, deck, img_size=224, margin=-20, plate_only=False):
    if plate_only:
        shift = 40
        cy, cx = image.shape[0]//2, image.shape[1]//2
        x, y = cx-((img_size)//2), cy-((img_size)//2)
        
        if deck == 1 or deck == 2:
            # taking the center of the imagewith  size (img_size * img_size)
            return image[y:y+img_size, x:x+img_size]
        elif deck == 3:
            # taking the center of the image but shifted on y-axis
            return image[y-shift:y+img_size-shift, x:x+img_size]
        elif deck == 4 or deck == 5:
            #taking the center of the image but shifted on x-axis
            return image[y:y+img_size, x+shift+50:x+img_size+shift+50]
        elif deck == 6:
            # taking the center of the image shifted on both y and x axis
            return image[y-shift:y+img_size-shift, x+shift*2:x+img_size+shift*2]
        else:
            raise Exception("Deck number out of range.")
    else:
        h, w = image.shape[0], image.shape[1]
        y = (w-h)//2
        return image[:, y+margin:y+h+margin]

def resize_and_to_array(img, img_size=224):
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img, 0)

def preprocess_img(img, img_size=img_size, deck=None, plate_only=False, crop_shift=-20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if plate_only:
        img = crop_image(img, plate_only=True, deck=deck)                  # <<<<<<<<<<<<<<<<<<<< CROP SQUARE MARGIN
    else:
        img = crop_image(img, margin=crop_shift, deck=None)
    
    # img = resize_and_to_array(img, img_size=img_size)
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

def load_finetuned_model(finetuned_model_path, conv_output_layer="global_average_pooling"):
    # Building embedding model based on the loaded finetuned.
    finetuned_model = tf.keras.models.load_model(finetuned_model_path)
    
    input = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
    preprocess_layer = tf.cast(input, tf.float32)
    preprocess_layer = tf.keras.applications.mobilenet.preprocess_input(preprocess_layer)

    x = finetuned_model(preprocess_layer)

    if conv_output_layer == "global_average_pooling":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif conv_output_layer == "global_max_pooling":
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif conv_output_layer == "flatten":
        x = tf.keras.layers.Flatten()(x)
    else:
        raise Exception("conv_output_layer is unknown.")

    mobnet_extractor = tf.keras.models.Model(inputs=preprocess_layer, outputs=x)

    return mobnet_extractor, finetuned_model

def load_pretrained(conv_output_layer="global_average_pooling"):
    # Loading pretrained tf model.
    MobNetSmall = tf.keras.applications.MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    for layer in MobNetSmall.layers[:]:
        layer.trainable = False

    # Building embedding model based on the loaded pretrained.
    input = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
    preprocess_layer = tf.cast(input, tf.float32)
    preprocess_layer = tf.keras.applications.mobilenet.preprocess_input(preprocess_layer)

    x = MobNetSmall(preprocess_layer)

    if conv_output_layer == "global_average_pooling":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif conv_output_layer == "global_max_pooling":
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif conv_output_layer == "flatten":
        x = tf.keras.layers.Flatten()(x)
    else:
        raise Exception("conv_output_layer is unknown.")

    mobnet_extractor = tf.keras.models.Model(inputs=preprocess_layer, outputs=x)

    return mobnet_extractor, MobNetSmall

def extract_weights(filepath, model_embedding, plate_only_crop=False, crop_shift=-20, deck=None):
    filenames = os.listdir(filepath)
    weights_matrix = np.expand_dims(np.empty(model_embedding.output_shape[1]), 1)
    support_images = list()
    labels = list()

    for imgfile in filenames:
        if os.path.isdir(filepath+imgfile):
            continue # skip directories

        img = cv2.imread(filepath+imgfile)

        imgarr = preprocess_img(img, plate_only=plate_only_crop, crop_shift=crop_shift, deck=deck)

        plt.imshow(np.squeeze(imgarr))
        plt.axis("off")
        plt.show()
        
        imgarr = tf.cast(imgarr, tf.int32)
        embeddings = model_embedding.predict(imgarr)[0]                                        # extracting embeddings
        embeddings = tf.nn.l2_normalize(embeddings)                                            # normalize embeddings
        embeddings = np.expand_dims(embeddings, 1)
        weights_matrix = np.append(weights_matrix, embeddings, 1)                              # creating the weights matrix with the embeddings
        support_images.append(imgarr)
        labels.append(imgfile.split('-')[0]) # +'-'+imgfile.split('-')[1])

    weights_matrix = np.delete(weights_matrix, 0, 1)                                           # remove the empty column
    weights_matrix = tf.constant_initializer(weights_matrix)
    num_instances = len(labels)
    
    return weights_matrix, support_images, labels, num_instances


def build_model(pretrained_model, num_instances, weights_matrix, labels, conv_output_layer="global_average_pooling"):
    input = tf.keras.layers.Input([img_size, img_size, 3], dtype=tf.uint8)
    preprocess_layer = tf.cast(input, tf.float32)
    preprocess_layer = tf.keras.applications.mobilenet.preprocess_input(preprocess_layer)

    x = pretrained_model(preprocess_layer)
    
    if conv_output_layer == "global_average_pooling":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif conv_output_layer == "global_max_pooling":
        x = tf.keras.layers.GlobalMaxPool2D()(x)
    elif conv_output_layer == "flatten":
        x = tf.keras.layers.Flatten()(x)
    else:
        raise Exception("conv_output_layer unknown.")

    x = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x), name='l2_norm_layer')(x)
    retrieval_output = tf.keras.layers.Dense(
            num_instances,
            kernel_initializer=weights_matrix,
            activation="linear",
            trainable=False,
            name='retrieval_layer')(x)

    class_id = set(labels)
    selection_layer_output = list()

    for ci in sorted(class_id):
        class_index = [i for i, c in enumerate(labels) if c == ci]
        x = AggregationLayer(class_index)(retrieval_output)
        selection_layer_output.append(x)

    concatenated_ouput = tf.stack(selection_layer_output, axis=1)

    model = tf.keras.models.Model(inputs=preprocess_layer, outputs=concatenated_ouput)
    return model

def tf_ImageDataGenerator(seed_num=1):
    np.random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = '0'
    tf.random.set_seed(seed_num)
    random.seed(seed_num)

    return tf.keras.preprocessing.image.ImageDataGenerator(
        brightness_range = [0.1, 1.9]
    )