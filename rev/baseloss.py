import os,sys
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from IPython import embed 





# write losses.
def loss_ccsa(xs, ys, xt, yt, margin=1.0):
    ce_loss = nn.CrossEntropyLoss()
    class_eq = (ys==yt).float()
    dist = F.pairwise_distance(xs, xt)
    loss = class_eq * dist.pow(2) + (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss


def loss_dsne(xs, ys, xt, yt, margin=1.0):
    '''
    xs, xt: features
    ys, yt: labels
    ''' 

    batch_size = xs.shape[0]
    embed_size = xs.shape[1]

    xs_rpt = xs.unsqueeze(0).expand(batch_size, batch_size, embed_size)
    xt_rpt = xt.unsqueeze(1).expand(batch_size, batch_size, embed_size)

    dists = ((xt_rpt - xs_rpt)**2).sum(dim = 2)

    yt_rpt = yt.unsqueeze(1).expand(batch_size, batch_size)
    ys_rpt = ys.unsqueeze(0).expand(batch_size, batch_size)

    y_same = (yt_rpt == ys_rpt)
    y_diff = (yt_rpt != ys_rpt)

    intra_cls_dists = dists * (y_same.float())
    inter_cls_dists = dists * (y_diff.float())


    max_dists = dists.max(dim = 1, keepdim=True)[0]
    max_dists = max_dists.expand(batch_size, batch_size)
    revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)


    # print(revised_inter_cls_dists)

    max_intra_cls_dist = intra_cls_dists.max(dim = 1)[0]
    min_inter_cls_dist = revised_inter_cls_dists.min(dim=1)[0]

    # print(max_intra_cls_dist)
    # print(min_inter_cls_dist)

    loss = F.relu(max_intra_cls_dist - min_inter_cls_dist + margin)

    return loss


def loss_dage(xs, ys, xt, yt):
    '''
    xs, xt: features
    ys, yt: labels
    ''' 

    phi = torch.cat([xs,xt],0).t()
    bs = ys.shape[0]

    W, Wp = make_weights(xs, xt, ys, yt, bs)

    D = torch.diag(W.sum(dim=1))
    Dp = torch.diag(Wp.sum(dim=1))

    L = D - W 
    Lp= Dp-Wp 

    phi_t = phi.t().contiguous()

    # print(phi.shape, L.shape, phi_t.shape)

    phi_L_phi_t = torch.matmul( phi, torch.matmul(L, phi_t) )
    phi_Lp_phi_t= torch.matmul( phi, torch.matmul(Lp, phi_t))

    loss = torch.trace(phi_L_phi_t) / (torch.trace(phi_Lp_phi_t) + 1e-11)

    return loss


def make_weights(xs, xt, ys, yt, batch_size):
    connect = connect_source_target
    W, Wp = connect(ys, yt, batch_size)
    return W.float(), Wp.float()
    # return tf.cast(W, dtype=DTYPE), tf.cast(Wp, dtype=DTYPE)


# ConnectionTypes
def connect_all(ys, yt, batch_size):
    N = 2 * batch_size
    y = torch.cat([ys, yt], dim=0)
    # yTe = tf.broadcast_to(tf.expand_dims(y, axis=1), shape=(N, N))
    # eTy = tf.broadcast_to(tf.expand_dims(y, axis=0), shape=(N, N))

    yTe = y.unsqueeze(1).expand(N,N)
    eTy = y.unsqueeze(0).expand(N,N)

    W = (yTe == eTy)
    Wp = (yTe != eTy)

    return W, Wp


def connect_source_target(ys, yt, batch_size, intrinsic=True, penalty=True):
    W, Wp = connect_all(ys, yt, batch_size)

    N = 2 * batch_size
    # tile_size = tf.repeat(batch_size, 2)
    tile_size = [batch_size, batch_size]

    oo = torch.zeros(tile_size, dtype=torch.bool).cuda()
    ii = torch.ones(tile_size, dtype=torch.bool).cuda()
    # ind = tf.concat([tf.concat([oo, ii], axis=0), tf.concat([ii, oo], axis=0)], axis=1)
    # oo = zeros(tf.repeat(N, 2), dtype=tf.bool)

    ind = torch.cat([torch.cat([oo, ii], dim=0), torch.cat([ii, oo], dim=0)], dim=1)
    oo = torch.zeros((N,N), dtype=torch.bool).cuda()


    if intrinsic:
        W = torch.where(ind, W.float(), oo.float())
    if penalty:
        Wp = torch.where(ind, Wp.float(), oo.float())

    return W, Wp












def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # assert n_samples!=1
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)


    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        if (label==i).sum()==0:
            continue
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class






def main():
    pass

    
    
    







if __name__ =="__main__":
    main()