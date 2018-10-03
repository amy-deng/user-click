### ranking metrics
import random
import math

def act_score_bag(true, pred):
    mapped = zip(true,pred)
    return mapped

def _DCG(x):
    s = 0.0
    k = 1.0
    for r in x:
        s += (2**r - 1.0)/math.log(1.0 + k)
        k += 1.0
    return s

def metric_ap(act_score_bag):
    random.shuffle(act_score_bag)
    if act_score_bag == None or len(act_score_bag) == 0:
        return -1.0
    ap = 0.0
    act_score_sorted = sorted(act_score_bag, key=lambda pair: pair[1], reverse=True)
    k0 = 0
    k1 = 0
    for label, val in act_score_sorted:
        if label > 0.0:
            k1 += 1
            ap += float(k1) / float(k0 + k1)
        else:
            k0 += 1
    if k1 > 0.0:
        ap /= float(k1)
        return ap
    else:
        return -1.0
    
def metric_rr(act_score_bag):
    random.shuffle(act_score_bag)
    if act_score_bag == None or len(act_score_bag) == 0:
        return -1.0
    rr = 0.0
    act_score_sorted = sorted(act_score_bag, key=lambda pair: pair[1], reverse=True)
    k = 1.0
    flag = False
    for label, score in act_score_sorted:
        if label > 0.0:
            rr = 1.0 / float(k)
            flag = True
            break
        k += 1.0
    if flag:
        return rr
    else:
        return -1.0
    
def metric_auc(act_score_bag):
    random.shuffle(act_score_bag)
    if act_score_bag == None or len(act_score_bag) == 0:
        return -1.0
    act_score_sorted = sorted(act_score_bag, key=lambda pair: pair[1], reverse=True)
    auc = 0.0
    k0 = 0
    k1 = 0
    tp0 = 0.0
    fp0 = 0.0
    tp1 = 1.0
    fp1 = 1.0
    P = 0
    N = 0
    for act, val in act_score_sorted:
        if act > 0.0:
            P += 1.0
        else:
            N += 1.0
    if P == 0:
        return -1.0
    if N == 0:
        return -1.0

    for act, val in act_score_sorted:
        if act > 0.0:
            k1 += 1
            tp1 = float(k1) / float(P)
            fp1 = float(k0) / float(N)
            auc += (fp1 - fp0) * (tp1 + tp0) / 2.0
            tp0 = tp1
            fp0 = fp1
        else:
            k0 += 1
    auc += 1.0 - fp1
    return auc

def metric_ndcg(act_score_bag):

    random.shuffle(act_score_bag)
    if act_score_bag == None or len(act_score_bag) == 0:
        return -1.0
    act_score_sorted = sorted(act_score_bag, key=lambda pair: pair[1], reverse=True)
    dcglist = [x[0] for x in act_score_sorted]
    val1 = _DCG(dcglist)
    act_score_sorted = sorted(act_score_bag, key=lambda pair: pair[0], reverse=True)
    idcglist = [x[0] for x in act_score_sorted]
    val0 = _DCG(idcglist)
    if val0 == 0.0:
        return -1.0
    return val1/val0
