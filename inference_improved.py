import cv2
import json
import numpy as np
import os
import os.path as osp
import time
import glob
import sys
import argparse
from datetime import datetime
from numpy.lib.function_base import append
import pandas as pd
import progressbar

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes
from eval.common import _get_detections, _get_annotations
from eval.pascal import Evaluate
from generators.csv_ import CSVGenerator
from utils.compute_overlap import compute_overlap
from classification_model import create_model as create_cls_model, improve_detections


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.gpu and parsed_args.batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))

    return parsed_args


def parse_args(args):
    """
    Parse the arguments.
    """
    today = datetime.now().strftime('%Y%m%d-%H%M%S')

    parser = argparse.ArgumentParser(description='Simple inference script.')
    parser.add_argument('ed_model_path', help='path to EfficientDet model.h5')
    parser.add_argument('cls_model_path', help='path to Classification model.h5')
    parser.add_argument('annotations_path', help='Path to CSV file containing annotations for inference.')
    parser.add_argument('classes_path', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--detect-quadrangle', help='If to detect quadrangle.', action='store_true', default=False)
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true')
    parser.add_argument('--evaluate', help='Calculate evaluation metrics too', action='store_true')

    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--output-dir', help='Dir to save the ouput images',
                        default='inferences/{}'.format(today))

    parsed_args = parser.parse_args(args)
    print(vars(parsed_args))
    return parsed_args


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[args.phi]

    # generator
    generator = CSVGenerator(
        args.annotations_path,
        args.classes_path,
        batch_size = 1,
        phi = args.phi,
        detect_quadrangle = args.detect_quadrangle,
    )

    # csv classes
    classes_df = pd.read_csv(args.classes_path, index_col=False, header=None, names=['class_name', 'id'])
    classes = {row['id']: row['class_name'] for _, row in classes_df.iterrows()}
    num_classes = len(classes)
    true_color = (0, 255, 0)
    false_color = (0, 0, 255)
    duplicate_color = (0, 255, 255)
    # colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

    _, ed_model = efficientdet(
        args.phi,
        num_classes=num_classes,
        weighted_bifpn=args.weighted_bifpn,
        detect_quadrangle=args.detect_quadrangle,
    )
    ed_model.load_weights(args.ed_model_path, by_name=True)

    crop_size = (200, 200)
    cls_model = create_cls_model((*crop_size, 3), num_classes, verbose=False)
    cls_model.load_weights(args.cls_model_path, by_name=False)

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    iou_threshold = 0.5
    score_threshold = 0.1
    max_detections = 250

    # if args.evaluate:
    #     e = Evaluate(
    #         generator,
    #         ed_model,
    #         iou_threshold=iou_threshold,
    #         score_threshold=score_threshold,
    #         max_detections=max_detections,
    #         save_path=None,
    #         tensorboard=None,
    #         weighted_average=True,
    #         verbose=1,
    #     )
    #     e.on_epoch_end(0, None)

    all_detections = _get_detections(generator, ed_model, score_threshold, max_detections)
    all_annotations = _get_annotations(generator)

    true_positives, false_positives, false_negatives = 0, 0, 0

    for i in progressbar.progressbar(range(generator.size()), prefix='Inferencing images: '):
        # load the image
        image = generator.load_image(i)

        all_detections[i] = improve_detections(image, all_detections[i], cls_model, crop_size, num_classes, rbc_threshold=0.8)

        # find boxes
        real_boxes = []
        real_scores = []
        real_labels = []
        real_colors = []
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        pred_colors = []

        for label in range(generator.num_classes()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            detected_annotations = []

            for d in detections:
                pred_boxes.append(d[:4])
                pred_scores.append(d[4])
                pred_labels.append(label)

                if annotations.shape[0] == 0:
                    pred_colors.append(false_color)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold:
                    if assigned_annotation not in detected_annotations:
                        true_positives += 1
                        pred_colors.append(true_color)
                        detected_annotations.append(assigned_annotation)
                    else:
                        if label != generator.num_classes() - 1:
                            false_positives += 1
                        pred_colors.append(duplicate_color)
                else:
                    if label != generator.num_classes() - 1:
                        false_positives += 1
                    pred_colors.append(false_color)

            false_negatives += len(annotations) - len(detected_annotations)

            for ai, a in enumerate(annotations):
                real_boxes.append(a[:4])
                real_scores.append(1)
                real_labels.append(label)
                if label == generator.num_classes() - 1:
                    real_colors.append(duplicate_color)
                elif ai in detected_annotations:
                    real_colors.append(true_color)
                else:
                    real_colors.append(false_color)

        # draw boxes on image
        src_image = image.copy()
        draw_boxes(src_image, real_boxes, real_scores, real_labels, real_colors, classes)
        draw_boxes(image, pred_boxes, pred_scores, pred_labels, pred_colors, classes)
        image = np.concatenate((src_image, image), axis=1)

        cv2.imwrite(osp.join(args.output_dir, osp.basename(generator.image_path(i))), image)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'Precision = {precision:.03f}')
    print(f'Recall = {recall:.03f}')
    print(f'F1 Score = {f1_score:.03f}')

if __name__ == '__main__':
    main()
