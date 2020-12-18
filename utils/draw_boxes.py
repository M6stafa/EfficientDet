import cv2


def draw_boxes(image, boxes, scores, labels, colors, classes):
    for b, l, s, color in zip(boxes, labels, scores, colors):
        class_id = int(l)
        class_name = classes[class_id]

        xmin, ymin, xmax, ymax = list(map(int, b))
        label = f'{class_name}-{s:.04f}'

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
