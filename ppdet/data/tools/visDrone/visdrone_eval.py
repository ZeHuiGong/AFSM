import mmcv
import numpy as np
import os
import time
from tools.convert_datasets.visDrone._visdrone_eval import compiou, createIntImg
from pycocotools.cocoeval import maskUtils


def visdrone_eval(resPath, gtPath, gtPkl):
    """ eval visdrone results"""
    start = time.time()
    allgt, alldet = loadAnnoRes(gtPath, resPath, gtPkl)
    loadAnnTime = time.time()
    print('loadAnnTime', loadAnnTime - start)

    # claculate average precision and recall over all 10 IoU thresholds
    # (i.e., [0.50:0.05:0.95]) of all object categories
    # AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500
    eval_res = calcAccuracy(allgt, alldet)
    print_summary(eval_res)

    return eval_res


def print_summary(eval_res):
    def _summarize(val, ap=1, iouThr=None, maxDets=100):
        iStr = ' {:<18} {} @[ IoU={:<9} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(0.50, 0.95) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        print(iStr.format(titleStr, typeStr, iouStr, maxDets, val))

    # AP
    _summarize(eval_res[0], maxDets=500)
    _summarize(eval_res[1], iouThr=0.50, maxDets=500)
    _summarize(eval_res[2], iouThr=0.75, maxDets=500)

    # AR
    _summarize(eval_res[3], ap=0, maxDets=1)
    _summarize(eval_res[4], ap=0, maxDets=10)
    _summarize(eval_res[4], ap=0, maxDets=100)
    _summarize(eval_res[6], ap=0, maxDets=500)


def calcAccuracy(allgt, alldet):
    """claculate average precision and recall over all 10 IoU thresholds
    (i.e., [0.5:0.05:0.95]) of all avaliable object categories"""
    AP = np.zeros([10, 10])
    AR = np.zeros([10, 10, 4])
    evalClass = set()
    numImgs = len(allgt)

    for idClass in range(10):
        print('evaluating object category {}/10'.format(idClass + 1))
        #  find the avaliable object categories
        for idImg in range(numImgs):
            gt = allgt[idImg]
            if np.any(gt[:, 5] == idClass + 1):
                evalClass.add(idClass)
        x = 0
        for thr in np.linspace(0.5, 0.95, 10):
            y = 0
            for maxDets in [1, 10, 100, 500]:
                gtMatch = []
                detMatch = []
                for idImg in range(numImgs):
                    gt = allgt[idImg]
                    det = alldet[idImg]
                    idxGtCurClass = np.where(gt[:, 5] == idClass + 1)[0]
                    idxDetCurClass = np.where(det[0:min(det.shape[0], maxDets), 5] == idClass + 1)[0]
                    gt0 = gt[idxGtCurClass, 0:5]
                    dt0 = det[idxDetCurClass, 0:5]
                    gt1, dt1 = evalRes(gt0, dt0, thr)
                    gtMatch.append(gt1[:, 4:5])
                    detMatch.append(dt1[:, 4:6])
                gtMatch = np.vstack(gtMatch)
                detMatch = np.vstack(detMatch)
                idrank = np.argsort(-detMatch[:, 0], axis=0)
                tp = np.cumsum(detMatch[idrank, 1] == 1)
                rec = tp / max(1, gtMatch.size)
                if rec.size > 0:
                    AR[idClass, x, y] = np.max(rec) * 100
                y += 1
            fp = np.cumsum(detMatch[idrank, 1] == 0)
            prec = tp / np.maximum(1, fp + tp)
            AP[idClass, x] = VOCap(rec, prec) * 100
            x += 1
    evalClass = list(evalClass)
    AP_all = AP[evalClass, :].mean()
    AP_50 = AP[evalClass, 0].mean()
    AP_75 = AP[evalClass, 5].mean()
    AR_1 = AR[evalClass, :, 0].mean()
    AR_10 = AR[evalClass, :, 1].mean()
    AR_100 = AR[evalClass, :, 2].mean()
    AR_500 = AR[evalClass, :, 3].mean()
    print('Evaluation Completed. The peformance of the detector is presented as follows.')

    return AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500


def VOCap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (np.ndarray).
        precision: The precision curve (np.ndarray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evalRes(gt0, dt0, thr, mul=0):
    """Evaluates detections against ground truth data
    % Uses modified Pascal criteria that allows for "ignore" regions. The
    % Pascal criteria states that a ground truth bounding box (gtBb) and a
    % detected bounding box (dtBb) match if their overlap area (oa):
    %  oa(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(union(gtBb,dtBb))
    % is over a sufficient threshold (typically .5). In the modified criteria,
    % the dtBb can match any subregion of a gtBb set to "ignore". Choosing
    % gtBb' in gtBb that most closely matches dtBb can be done by using
    % gtBb'=intersect(dtBb,gtBb). Computing oa(gtBb',dtBb) is equivalent to
    %  oa'(gtBb,dtBb) = area(intersect(gtBb,dtBb)) / area(dtBb)
    % For gtBb set to ignore the above formula for oa is used.
    %
    % Highest scoring detections are matched first. Matches to standard,
    % (non-ignore) gtBb are preferred. Each dtBb and gtBb may be matched at
    % most once, except for ignore-gtBb which can be matched multiple times.
    % Unmatched dtBb are false-positives, unmatched gtBb are false-negatives.
    % Each match between a dtBb and gtBb is a true-positive, except matches
    % between dtBb and ignore-gtBb which do not affect the evaluation criteria.
    %
    % In addition to taking gt/dt results on a single image, evalRes() can take
    % cell arrays of gt/dt bbs, in which case evaluation proceeds on each
    % element. Use bbGt>loadAll() to load gt/dt for multiple images.
    %
    % Each gt/dt output row has a flag match that is either -1/0/1:
    %  for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
    %  for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]
    %
    % USAGE
    %  [gt, dt] = bbGt( 'evalRes', gt0, dt0, [thr], [mul] )
    %
    % INPUTS
    %  gt0  - [mx5] ground truth array with rows [x y w h ignore]
    %  dt0  - [nx5] detection results array with rows [x y w h score]
    %  thr  - [.5] the threshold on oa for comparing two bbs
    %  mul  - [0] if true allow multiple matches to each gt
    %
    % OUTPUTS
    %  gt   - [mx5] ground truth results [x y w h match]
    %  dt   - [nx6] detection results [x y w h score match]
    %
    % EXAMPLE
    %
    % See also bbGt, bbGt>compOas, bbGt>loadAll"""

    # if gt0 and dt0 are cell arrays run on each element in turn
    if isinstance(gt0, list) and isinstance(dt0, list):
        n = len(gt0)
        assert len(dt0) == n
        gt = [None for _ in range(n)]
        dt = [None for _ in range(n)]
        for i in range(n):
            gt[i], dt[i] = evalRes(gt0[i], dt0[i], thr, mul)
        return gt, dt
    # check inputs
    gt0 = np.zeros([0, 5]) if gt0.shape[0] == 0 else gt0
    dt0 = np.zeros([0, 5]) if dt0.shape[0] == 0 else dt0
    assert gt0.shape[1] == 5
    assert dt0.shape[1] == 5
    nd, ng = dt0.shape[0], gt0.shape[0]

    # sort dt highest score first, sort gt ignore last
    dt0 = dt0[np.argsort(-dt0[:, 4]), :]
    gt0 = gt0[np.argsort(gt0[:, 4]), :]
    gt, dt = gt0, np.hstack([dt0, np.zeros([nd, 1])])
    gt[:, 4] = -gt[:, 4]
    # Attempt to match each (sorted) dt to each (sorted) gt
    # ious = compiou(dt[:, 0:4].astype(np.float32), gt[:, 0:4].astype(np.float32), gt[:, 4] == -1)
    ious = maskUtils.iou(dt[:, 0:4].astype(np.float32), gt[:, 0:4].astype(np.float32), gt[:, 4] == -1)
    for d in range(nd):
        bstOa = thr
        bstg = 0
        bstm = 0  # % info about best match so far
        for g in range(ng):
            # if this gt already matched, continue to next gt
            m = gt[g, 4]
            if m == 1 and not mul: continue
            # if dt already matched, and on ignore gt, nothing more to do
            if bstm != 0 and m == -1: break
            # compute overlap area, continue to next gt unless better match made
            if ious[d, g] < bstOa: continue
            # match successful and best so far, store appropriately
            bstOa = ious[d, g]
            bstg = g
            if m == 0:
                bstm = 1
            else:  # false positive
                bstm = -1
        g = bstg
        m = bstm
        #  store type of match for both dt and gt
        if m == -1:
            dt[d, 5] = m
        elif m == 1:
            gt[g, 4] = m
            dt[d, 5] = m
    return gt, dt


# def compiou(dt, gt, ig=None):
#     """% Computes (modified) overlap area between pairs of bbs.
#     %
#     % Uses modified Pascal criteria with "ignore" regions. The overlap area
#     % (oa) of a ground truth (gt) and detected (dt) bb is defined as:
#     %  oa(gt,dt) = area(intersect(dt,dt)) / area(union(gt,dt))
#     % In the modified criteria, a gt bb may be marked as "ignore", in which
#     % case the dt bb can can match any subregion of the gt bb. Choosing gt' in
#     % gt that most closely matches dt can be done using gt'=intersect(dt,gt).
#     % Computing oa(gt',dt) is equivalent to:
#     %  oa'(gt,dt) = area(intersect(gt,dt)) / area(dt)
#     %
#     % USAGE
#     %  oa = bbGt( 'compOas', dt, gt, [ig] )
#     %
#     % INPUTS
#     %  dt       - [mx4] detected bbs
#     %  gt       - [nx4] gt bbs
#     %  ig       - [nx1] 0/1 ignore flags (0 by default)
#     %
#     % OUTPUTS
#     %  oas      - [m x n] overlap area between each gt and each dt bb
#     """
#     m, n = dt.shape[0], gt.shape[0]
#     ious = np.zeros([m, n])
#     ig = np.zeros([n, 1], dtype=np.bool) if ig is None else ig
#     de = dt[:, [0, 1]] + dt[:, [2, 3]]  # x2, y2
#     da = dt[:, 2] * dt[:, 3]  # area
#     ge = gt[:, [0, 1]] + gt[:, [2, 3]]  # x2, y2
#     ga = gt[:, 2] * gt[:, 3]  # area
#     for i in range(m):      # loop through det boxes
#         for j in range(n):  # loop through gt boxes
#             w = min(de[i, 0], ge[j, 0]) - max(dt[i, 0], gt[j, 0])
#             h = min(de[i, 1], ge[j, 1]) - max(dt[i, 1], gt[j, 1])
#             if w <= 0 or h <= 0:
#                 continue
#             t = w * h
#             if ig[j]:
#                 u = da[i]
#             else:
#                 u = da[i] + ga[j] - t
#             ious[i, j] = t / u
#     return ious


def loadAnnoRes(gtPath, resPath, gtPkl):
    """process the annotations and groundtruth
       :return allgt alldet:list[np.ndarray]
    """
    gtpkl = mmcv.load(gtPkl)
    allgt = []
    alldet = []
    for anno in gtpkl:
        nameImg = anno['filename']
        imgHeight = anno['height']
        imgWidth = anno['width']
        # print(nameImg)
        oldgt = loadtxt(os.path.join(gtPath, nameImg[:-3] + 'txt'))
        olddet = loadtxt(os.path.join(resPath, nameImg[:-3] + 'txt'))
        # remove the objects in ignored regions or labeled as others
        # newgt, det = dropObjectsInIgr(oldgt, olddet, imgHeight, imgWidth)
        newgt = oldgt
        det = olddet
        gt = np.copy(newgt)
        gt[newgt[:, 4] == 0, 4] = 1
        gt[newgt[:, 4] == 1, 4] = 0
        allgt.append(gt)
        alldet.append(det)
    return allgt, alldet


def loadtxt(txt_file):
    lines = mmcv.list_from_file(txt_file)
    if len(lines) == 0:
        return np.zeros([0, 8], dtype=np.int)
    lines = [list(map(int, line.split(',')[:-1])) for line in lines]
    lines = np.array(lines, dtype=np.int)
    return lines


# def createIntImg(img):
#     height, width = img.shape
#     img[:, 0] = np.cumsum(img[:, 0], axis=0)
#     img[0, :] = np.cumsum(img[0, :], axis=0)
#     shift = [img.copy() for _ in range(3)]
#     shift[0][:height-1, :] = img[1:height, :]  # up
#     shift[1][:, :width-1] = img[:, 1:width]  # left
#     shift[2][:height-1, :width-1] = img[1:height, 1:width]  # up left
#     img = img + shift[0] + shift[1] - shift[2]
#     return img


def dropObjectsInIgr(gt, det, imgHeight, imgWidth):
    """drop annotations and detection in ignored region"""
    # parse objects
    idxFr = np.where(gt[:, 5] != 0)[0]
    curgt = gt[idxFr]

    # parse ignored regions
    idxIgr = np.where(gt[:, 5] == 0)[0]
    igrRegion = np.maximum(0, gt[idxIgr, 0:4])
    numIgr = idxIgr.shape[0]
    if numIgr > 0:
        igrMap = np.zeros([imgHeight, imgWidth], dtype=np.int32)
        for j in range(numIgr):
            igrMap[igrRegion[j, 1]:np.minimum(imgHeight-1, igrRegion[j, 1]+igrRegion[j, 3]),
            igrRegion[j, 0]:np.minimum(imgWidth-1, igrRegion[j, 0] + igrRegion[j, 2])] = 1
        intIgrMap = createIntImg(igrMap)
        # ignore gt
        x = np.maximum(0, np.minimum(imgWidth-1, curgt[:, 0]))
        y = np.maximum(0, np.minimum(imgHeight - 1, curgt[:, 1]))
        w = curgt[:, 2]
        h = curgt[:, 3]
        tl = intIgrMap[y, x]
        tr = intIgrMap[y, np.minimum(imgWidth - 1, x + w)]
        bl = intIgrMap[np.maximum(0, np.minimum(imgHeight - 1, y + h)), x]
        br = intIgrMap[np.maximum(0, np.minimum(imgHeight - 1, y + h)),
                       np.minimum(imgWidth - 1, x + w)]
        igrVal = (tl + br - tr - bl) / (h * w)
        idxLeftGt = np.where(igrVal < 0.5)[0]
        curgt = curgt[idxLeftGt]

        # ignore det
        x = np.maximum(0, np.minimum(imgWidth - 1, det[:, 0]))
        y = np.maximum(0, np.minimum(imgHeight - 1, det[:, 1]))
        w = det[:, 2]
        h = det[:, 3]
        tl = intIgrMap[y, x]
        tr = intIgrMap[y, np.minimum(imgWidth - 1, x + w)]
        bl = intIgrMap[np.maximum(0, np.minimum(imgHeight - 1, y + h)), x]
        br = intIgrMap[np.maximum(0, np.minimum(imgHeight - 1, y + h)),
                       np.minimum(imgWidth - 1, x + w)]
        igrVal = (tl + br - tr - bl) / (h * w)
        idxLeftDet = np.where(igrVal < 0.5)[0]
        det = det[idxLeftDet]

    return curgt, det


if __name__ == '__main__':
    resPath = '../../../work_dirs/visdrone2019/centernet_hourglass-52_1x-06-16/results/result-hg52-bst'
    gtPath = '/media/jp/新加卷/ZEHUI_DATA/Dataset/VisDrone2019/VisDrone2019-DET-val/annotations'
    gtPkl = '../../../data/visdrone2019/annotations/visdrone2019-val.pkl'
    visdrone_eval(resPath, gtPath, gtPkl)

