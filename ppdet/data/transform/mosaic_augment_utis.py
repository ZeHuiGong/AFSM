import random
import numpy as np
import cv2


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(gt_bbox, dx, dy, sx, sy, net_w, net_h, flip=0):
    """
    1. correct box coordinate with the given box (dx, dy, sx, sy)
    2. filter out boxes that are fall out of the region
    3. transform coordinates from crop region to input image system
    Args:
        sample (dict): a dict that store information of a image
    """
    if gt_bbox.shape[0] == 0:
        return gt_bbox, 10000
    np.random.shuffle(gt_bbox)
    gt_bbox[:, 0] -= dx
    gt_bbox[:, 2] -= dx
    gt_bbox[:, 1] -= dy
    gt_bbox[:, 3] -= dy

    gt_bbox[:, 0] = np.clip(gt_bbox[:, 0], 0, sx)
    gt_bbox[:, 2] = np.clip(gt_bbox[:, 2], 0, sx)

    gt_bbox[:, 1] = np.clip(gt_bbox[:, 1], 0, sy)
    gt_bbox[:, 3] = np.clip(gt_bbox[:, 3], 0, sy)

    out_box = list(np.where(((gt_bbox[:, 1] == sy) & (gt_bbox[:, 3] == sy)) |
                            ((gt_bbox[:, 0] == sx) & (gt_bbox[:, 2] == sx)) |
                            ((gt_bbox[:, 1] == 0) & (gt_bbox[:, 3] == 0)) |
                            ((gt_bbox[:, 0] == 0) & (gt_bbox[:, 2] == 0)))[0])
    list_box = list(range(gt_bbox.shape[0]))
    for i in out_box:
        list_box.remove(i)
    gt_bbox = gt_bbox[list_box]

    if gt_bbox.shape[0] == 0:
        return gt_bbox, 10000

    min_w_h = np.array([gt_bbox[:, 2] - gt_bbox[:, 0], gt_bbox[:, 3] - gt_bbox[:, 1]]).min()

    gt_bbox[:, 0] *= (net_w / sx)
    gt_bbox[:, 2] *= (net_w / sx)
    gt_bbox[:, 1] *= (net_h / sy)
    gt_bbox[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - gt_bbox[:, 0]
        gt_bbox[:, 0] = net_w - gt_bbox[:, 2]
        gt_bbox[:, 2] = temp

    return gt_bbox, min_w_h


def rect_intersection(a, b):
    """a, b: two boxes [x1, y1, x2, y2]"""
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, truth, w, h, pleft, ptop, swidth, sheight, flip=0,
                            dhue=0, dsat=1, dexp=1, gaussian_noise=0, blur=0):
    """
    1. get final crop_box that we want to crop
    2. crop image patch from original image
    3. execute image augmentation to cropped image patch
    """
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集
        # (x1, y1, x2, y2)
        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]:
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            cropped = np.zeros([sheight, swidth, 3])
            # cropped 初始化为img 像素的均值， 相当于global average pooling
            cropped[:, :, ] = np.mean(img, axis=(0, 1))

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            # if blur == 1:
            #     dst = cv2.GaussianBlur(sized, (17, 17), 0)
            #     # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            # else:
            ksize = int((blur / 2) * 2 + 1)
            dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            # if blur == 1:
            #     img_rect = [0, 0, sized.shape[1], sized.shape[0]]
            #     for b in truth:
            #         left = b[0] * sized.shape[1]
            #         width = (b[2] - b[0]) * sized.shape[1]
            #         top = b[1] * sized.shape[0]
            #         height = (b[3] - b[1]) * sized.shape[0]
            #         roi = [left, top, width, height]
            #         roi = rect_intersection(roi, img_rect)
            #         dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
            #                                                                     roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    """pad img to the corresponding region in out_img
    Args:
        out_img (np.array): output image with shape (mosaic_w, mosaic_h)
        img: (mosaic_w, mosaic_h) i_mixup-th img that need to be padded in out_img
        cut_x, cut_y: intersect-point of four img in out_img
        left_shift, right_shift, top_shift, bot_shift:
    """
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def show_img(sample):
    im = sample['image'].astype(np.uint8)
    gt_bbox = sample['gt_bbox'].astype(np.int32)
    gt_class = sample['gt_class']

    for i in range(gt_bbox.shape[0]):
        left_top = (gt_bbox[i, 0], gt_bbox[i, 1])
        right_bottom = (gt_bbox[i, 2], gt_bbox[i, 3])
        cv2.rectangle(im, left_top, right_bottom, color=(255, 0, 0), thickness=2)
        label_text = 'cls {}'.format(gt_class[i, 0])
        cv2.putText(im, label_text, (gt_bbox[i, 0], gt_bbox[i, 1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 1, color=(0, 255, 0))
    cv2.imshow('image', im)
    cv2.waitKey(0)


def corner_crop_image(image, center, size):
        cty, ctx = center
        height, width = size
        im_height, im_width = image.shape[0:2]
        cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

        x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
        y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

        left, right = ctx - x0, x1 - ctx
        top, bottom = cty - y0, y1 - cty

        cropped_cty, cropped_ctx = height // 2, width // 2
        y_slice = slice(cropped_cty - top, cropped_cty + bottom)
        x_slice = slice(cropped_ctx - left, cropped_ctx + right)
        cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_cty - top,
            cropped_cty + bottom,
            cropped_ctx - left,
            cropped_ctx + right
        ], dtype=np.float32)

        offset = np.array([
            max(0, cty - height // 2),
            max(0, ctx - width // 2)
        ])

        return cropped_image, border, offset
