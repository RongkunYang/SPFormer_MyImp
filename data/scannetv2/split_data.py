import os
import shutil

# split scans specified in scannetv2_train/val/test.txt

splits = ['train','val','test']

for split in splits:
    print('processing ',split)
    f_name = f'scannetv2_{split}.txt'
    f = open(f_name, 'r')
    scans = f.readlines()
    os.makedirs(split, exist_ok=True)
    for scan_name in scans:
        scan = scan_name.strip()
        if split == 'test':
            src = '/home/yrk/Desktop/program/datasets/scannet/scans_test/{}/{}_vh_clean_2.ply'.format(scan, scan)
            dest = f'{split}/{scan}_vh_clean_2.ply'
            shutil.copyfile(src,dest)
        else:
            src = '/home/yrk/Desktop/program/datasets/scannet/scans/{}/{}_vh_clean_2.ply'.format(scan, scan)
            dest = '{}/{}_vh_clean_2.ply'.format(split, scan)
            shutil.copyfile(src, dest)

            src = '/home/yrk/Desktop/program/datasets/scannet/scans/{}/{}_vh_clean_2.labels.ply'.format(scan, scan)
            dest = '{}/{}_vh_clean_2.labels.ply'.format(split, scan)
            shutil.copyfile(src, dest)

            src = '/home/yrk/Desktop/program/datasets/scannet/scans/{}/{}_vh_clean_2.0.010000.segs.json'.format(scan,
                                                                                                                scan)
            dest = '{}/{}_vh_clean_2.0.010000.segs.json'.format(split, scan)
            shutil.copyfile(src, dest)

            src = '/home/yrk/Desktop/program/datasets/scannet/scans/{}/{}.aggregation.json'.format(scan, scan)
            dest = '{}/{}.aggregation.json'.format(split, scan)
            shutil.copyfile(src, dest)
print('done')
