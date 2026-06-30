"""
YOLOv3 in TensorFlow/Keras — model, loss, Darknet weight loader, and data pipeline.

Architecture follows the well-established yolov3-tf2 design (Darknet-53 backbone +
3-scale FPN heads). Kept as a library; training/export live in the sibling scripts.

Anchors are normalized to image size (resolution independent). Inputs are 3-channel;
the data pipeline tiles single-channel grayscale PNGs to 3 channels so the pretrained
RGB backbone weights stay valid.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, Concatenate, Conv2D, Input, Lambda, LeakyReLU,
    UpSampling2D, ZeroPadding2D, BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

# Anchors come from scripts/anchors.json (k-means) if present, else stock COCO anchors.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from anchors_config import load_anchors as _load_anchors
yolo_anchors, _masks = _load_anchors()
yolo_anchor_masks = np.array(_masks)


# ----------------------------- model blocks -----------------------------
def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = "same"
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    return Add()([prev, x])


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)
    x = x_36 = DarknetBlock(x, 256, 8)
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    def _conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return _conv


def YoloOutput(filters, anchors, classes, name=None):
    def _out(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda t: tf.reshape(
            t, (-1, tf.shape(t)[1], tf.shape(t)[2], anchors, classes + 5)))(x)
        return Model(inputs, x, name=name)(x_in)
    return _out


def yolo_boxes(pred, anchors, classes):
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # for loss
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, objectness, class_probs, pred_box


def yolo_raw(outputs, classes):
    """Concat decoded predictions across scales into one (batch, N, 4+1+classes) tensor.
    Layout per row: [x1, y1, x2, y2 (normalized), objectness, *class_probs].
    NMS-free so the graph converts cleanly to OpenVINO IR (NMS done in postprocessing)."""
    bb, sc = [], []
    for bbox, objectness, class_probs in outputs:
        bb.append(tf.reshape(bbox, (tf.shape(bbox)[0], -1, 4)))
        conf = tf.reshape(objectness, (tf.shape(objectness)[0], -1, 1))
        cls = tf.reshape(class_probs, (tf.shape(class_probs)[0], -1, classes))
        sc.append(tf.concat([conf, cls], axis=-1))
    return tf.concat([tf.concat(bb, axis=1), tf.concat(sc, axis=1)], axis=-1)


def YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks,
           classes=80, training=False):
    x = inputs = Input([size, size, channels], name="input")
    x_36, x_61, x = Darknet(name="yolo_darknet")(x)
    x = YoloConv(512, name="yolo_conv_0")(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name="yolo_output_0")(x)
    x = YoloConv(256, name="yolo_conv_1")((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name="yolo_output_1")(x)
    x = YoloConv(128, name="yolo_conv_2")((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name="yolo_output_2")(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name="yolov3")

    boxes_0 = Lambda(lambda t: yolo_boxes(t, anchors[masks[0]], classes))(output_0)
    boxes_1 = Lambda(lambda t: yolo_boxes(t, anchors[masks[1]], classes))(output_1)
    boxes_2 = Lambda(lambda t: yolo_boxes(t, anchors[masks[2]], classes))(output_2)

    # OpenVINO-friendly inference graph: decoded boxes+scores, NMS done on the host.
    out = Lambda(lambda t: yolo_raw(t, classes))(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    return Model(inputs, out, name="yolov3")


# ----------------------------- loss -----------------------------
def broadcast_iou(box_1, box_2):
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area + 1e-9)


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def loss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        obj_mask = tf.squeeze(true_obj, -1)
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(
                x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask), fn_output_signature=tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), -1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), -1)
        obj_entropy = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_entropy + (1 - obj_mask) * ignore_mask * obj_entropy
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return xy_loss + wh_loss + obj_loss + class_loss
    return loss


# ----------------------------- darknet weight loader -----------------------------
def load_darknet_weights(model, weights_file):
    """Parse the official Darknet yolov3.weights and assign into a classes=80 model."""
    with open(weights_file, "rb") as wf:
        np.fromfile(wf, dtype=np.int32, count=5)  # header
        layers = ["yolo_darknet", "yolo_conv_0", "yolo_output_0", "yolo_conv_1",
                  "yolo_output_1", "yolo_conv_2", "yolo_output_2"]
        for layer_name in layers:
            sub = model.get_layer(layer_name)
            for i, layer in enumerate(sub.layers):
                if not layer.name.startswith("conv2d"):
                    continue
                batch_norm = None
                if i + 1 < len(sub.layers) and sub.layers[i + 1].name.startswith("batch_norm"):
                    batch_norm = sub.layers[i + 1]
                filters = layer.filters
                size = layer.kernel_size[0]
                try:
                    in_dim = layer.input.shape[-1]      # Keras 3
                except AttributeError:
                    in_dim = layer.input_shape[-1]      # Keras 2 fallback
                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(
                    wf, dtype=np.float32, count=int(np.prod(conv_shape)))
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
                if batch_norm is None:
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if hasattr(model, "layers"):
        for l in model.layers:
            freeze_all(l, frozen)


# ----------------------------- data pipeline -----------------------------
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / tf.cast(grid_size, tf.float32)), tf.int32)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    outs = []
    grid_size = size // 32
    anchors = tf.cast(anchors, tf.float32)
    area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    y_train = tf.concat([y_train, anchor_idx], axis=-1)
    for masks in anchor_masks:
        outs.append(transform_targets_for_output(y_train, grid_size, masks))
        grid_size *= 2
    return tuple(outs)


def _decode_image(path, size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.grayscale_to_rgb(img)          # tile gray -> 3ch for RGB backbone
    img = tf.image.resize(img, (size, size))
    return tf.cast(img, tf.float32) / 255.0


def _load_label(label_path, max_boxes):
    """Read a YOLO txt -> (max_boxes, 5) as (x1,y1,x2,y2,class)."""
    def _py(p):
        p = p.numpy().decode()
        rows = []
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c, cx, cy, bw, bh = map(float, line.split()[:5])
                    rows.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, c])
        except FileNotFoundError:
            pass
        arr = np.zeros((max_boxes, 5), np.float32)
        if rows:
            rows = np.array(rows[:max_boxes], np.float32)
            arr[:len(rows)] = rows
        return arr
    out = tf.py_function(_py, [label_path], tf.float32)
    out.set_shape((max_boxes, 5))
    return out


def _rot90_boxes(boxes, k):
    """Rotate normalized (x1,y1,x2,y2,cls) boxes to match tf.image.rot90 (CCW, k×90°)."""
    x1, y1, x2, y2, cls = tf.unstack(boxes, axis=1)
    branches = {
        0: lambda: tf.stack([x1, y1, x2, y2], 1),
        1: lambda: tf.stack([y1, 1 - x2, y2, 1 - x1], 1),     # 90° CCW
        2: lambda: tf.stack([1 - x2, 1 - y2, 1 - x1, 1 - y1], 1),  # 180°
        3: lambda: tf.stack([1 - y2, x1, 1 - y1, x2], 1),     # 270° CCW
    }
    return tf.concat([tf.switch_case(k, branches), cls[:, None]], axis=1)


def augment(img, boxes):
    """Box-exact online augmentation for grayscale PCB: h/v flips, 90° rotation,
    brightness/contrast. Padding rows (all-zero boxes) are restored to zero so flips
    don't turn them into spurious detections."""
    valid = boxes[:, 2] > boxes[:, 0]                         # real boxes have x2 > x1

    def hflip():
        x1, y1, x2, y2, c = tf.unstack(boxes, axis=1)
        return tf.image.flip_left_right(img), tf.stack([1 - x2, y1, 1 - x1, y2, c], 1)
    img2, boxes2 = tf.cond(tf.random.uniform([]) < 0.5, hflip, lambda: (img, boxes))

    def vflip():
        x1, y1, x2, y2, c = tf.unstack(boxes2, axis=1)
        return tf.image.flip_up_down(img2), tf.stack([x1, 1 - y2, x2, 1 - y1, c], 1)
    img3, boxes3 = tf.cond(tf.random.uniform([]) < 0.5, vflip, lambda: (img2, boxes2))

    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img4 = tf.image.rot90(img3, k)
    boxes4 = _rot90_boxes(boxes3, k)

    img5 = tf.image.random_brightness(img4, 0.1)
    img5 = tf.image.random_contrast(img5, 0.85, 1.15)
    img5 = tf.clip_by_value(img5, 0.0, 1.0)

    boxes5 = tf.where(valid[:, None], boxes4, tf.zeros_like(boxes4))
    return img5, boxes5


def make_dataset(split_dir, anchors, anchor_masks, size, classes,
                 batch_size, max_boxes=100, shuffle=True, augment_data=False):
    import pathlib
    img_dir = pathlib.Path(split_dir) / "images"
    paths = sorted(str(p) for p in img_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    lbls = [p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for p in paths]
    ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 2000))

    def _map(img_path, lbl_path):
        img = _decode_image(img_path, size)
        boxes = _load_label(lbl_path, max_boxes)   # (max_boxes, 5)
        if augment_data:
            img, boxes = augment(img, boxes)
        return img, boxes

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (x, transform_targets(y, anchors, anchor_masks, size)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), len(paths)
