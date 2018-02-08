"""Test utilities."""
from __future__ import print_function


def thrdiff(tree, fpthresh, skthresh, sknl, sknr, fpn=0, skn=0, parent=-1):
    """Threshold difference finder between scikit and forpy trees."""
    if tree[fpn][0] == 0 and tree[fpn][1] == 0:
        # leaf
        if skthresh[skn] != -2.:
            print('forpy leaf, but not skleaf', fpn, skn)
        return
    if fpthresh[fpn] != skthresh[skn]:
        print('fpn:', fpn, 'fppn:', parent, 'skn:', skn, 'fpthresh:',
              fpthresh[fpn], 'skthresh:', skthresh[skn])
    else:
        lfpid = tree[fpn][0]
        rfpid = tree[fpn][1]
        lskid = sknl[skn]
        rskid = sknr[skn]
        thrdiff(
            tree,
            fpthresh,
            skthresh,
            sknl,
            sknr,
            fpn=lfpid,
            skn=lskid,
            parent=fpn)
        thrdiff(
            tree,
            fpthresh,
            skthresh,
            sknl,
            sknr,
            fpn=rfpid,
            skn=rskid,
            parent=fpn)
