from os import listdir
import os.path as osp
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.backend as KB
import tensorflow.keras.layers as KL
from skimage.transform import resize
from skimage.io import imread
import imgaug.augmenters as iaa

from utils.compute_overlap import compute_overlap


def get_image_crops(generator, detections, annotations, image_index, crop_size, iou_threshold):
    crops = []
    labels = []

    image = generator.load_image(image_index)
    detected_annotations = []

    img_detections = np.vstack(detections[image_index])
    img_annotations = np.vstack(annotations[image_index])  # Flatten img_annotations

    annotations_label = np.zeros((img_annotations.shape[0],), dtype=np.int)
    start_index = 0
    for label in range(generator.num_classes()):
        num_annotations = len(annotations[image_index][label])
        if num_annotations == 0: continue
        annotations_label[start_index:start_index + num_annotations] = label
        start_index += num_annotations

    # Sort boxes descending by score
    scores = img_detections[:, 4]
    index_sorted = np.argsort(-scores)
    boxes = img_detections[index_sorted, :4]

    for box in boxes:
        xmin, ymin, xmax, ymax = list(map(int, box))
        crop = image[ymin:ymax, xmin:xmax]

        crop = resize(crop, (crop_size[1], crop_size[0]), preserve_range=True)

        # Find the true label
        overlaps = compute_overlap(np.expand_dims(box, axis=0), img_annotations)
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]

        if max_overlap >= iou_threshold:
            if assigned_annotation not in detected_annotations:
                detected_annotations.append(assigned_annotation)
                if annotations_label[assigned_annotation] == generator.name_to_label('difficult'):
                    continue  # skip this crop, because we don't know the label
                labels.append(int(annotations_label[assigned_annotation]))
            else:  # Duplicate
                if max_overlap >= iou_threshold + 0.15:
                    labels.append(int(annotations_label[assigned_annotation]))
                else:
                    labels.append(generator.num_classes() - 1)
        else:  # No enough overlap with annotations
            labels.append(generator.num_classes() - 1)

        crops.append(crop)

    return np.array(crops, dtype='uint8'), np.array(labels, dtype='uint8')


def create_model(input_shape, num_classes, verbose=True):
    def set_trainable(model, after_layer):
        trainable = False
        for layer in model.layers:
            layer.trainable = trainable

            if not trainable and layer.name == after_layer:
                trainable = True


    inputs = KL.Input(shape=input_shape)
    x = inputs
    x = tf.cast(x, tf.float32)

    core_kwargs = dict(include_top=False, weights='imagenet', input_shape=(*x.shape[1:],), pooling=None)

    x = K.applications.xception.preprocess_input(x)
    core = K.applications.Xception(**core_kwargs)
    set_trainable(core, 'block10_sepconv3_bn')

    x = core(x, training=False)

    x = KL.GlobalAvgPool2D(name='GlobalAvgPool2D')(x)

    x = KL.Dense(1024)(x)
    x = KL.BatchNormalization()(x)
    x = KL.ReLU()(x)
    x = KL.Dropout(0.4)(x)

    outputs = KL.Dense(num_classes, KL.Activation('softmax'), name='output')(x)

    model = K.Model(inputs, outputs, name='Model')

    if verbose:
        K.utils.plot_model(model, to_file='classification_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        model.summary()

    return model


class CropSequence(K.utils.Sequence):
    def __init__(self, base_path, num_classes, batch_size, augmenter=None, calc_weights=False, ignore_labels=[]):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augmenter = augmenter

        self.x = []
        self.y = []
        self.objects_per_class = np.zeros((num_classes,), dtype=np.int)

        for label in range(num_classes):
            if label in ignore_labels: continue

            label_path = osp.join(base_path, str(label))
            label_images = [imread(osp.join(label_path, f)) for f in listdir(label_path) if osp.isfile(osp.join(label_path, f))]

            self.x += label_images
            self.y += [label for _ in range(len(label_images))]
            self.objects_per_class[label] = len(label_images)

        self.x = np.array(self.x, dtype='uint8')
        self.y = np.array(self.y, dtype='uint8')

        self.indexes = np.arange(0, len(self.x), 1, dtype=np.int)
        np.random.shuffle(self.indexes)

        self.class_weights = np.ones((num_classes,), dtype=np.float)
        if calc_weights:
            non_zero_idx = self.objects_per_class > 0
            self.class_weights[non_zero_idx] = np.max(self.objects_per_class) / self.objects_per_class[non_zero_idx]


    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))


    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_idx]
        batch_y = self.y[batch_idx]

        if self.augmenter is not None:
            batch_x = self.augmenter(images=batch_x)

        batch_y = np.eye(self.num_classes)[batch_y]

        return batch_x, batch_y


def improve_detections(image, detections, cls_model, crop_size, num_classes):
    detections_stack = np.vstack(detections)
    boxes = detections_stack[:, :4]
    labels = np.zeros((boxes.shape[0],))
    scores = np.zeros((boxes.shape[0],))
    true_positives = np.full((boxes.shape[0],), True)

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = list(map(int, box))
        crop = image[ymin:ymax, xmin:xmax]
        crop = resize(crop, (crop_size[1], crop_size[0]), preserve_range=True)

        y_pred = cls_model.predict(crop[np.newaxis, ...])[0]
        label = np.argmax(y_pred)

        if label == num_classes - 1:
            true_positives[i] = False
            continue

        labels[i] = label
        scores[i] = y_pred[label]

    boxes = boxes[true_positives]
    labels = labels[true_positives]
    scores = scores[true_positives]

    new_detections = [None for _ in range(num_classes)]
    for label in range(num_classes):
        idx = labels == label
        new_detections[label] = np.concatenate([boxes[idx], np.expand_dims(scores[idx], axis=1)], axis=1)

    return new_detections
