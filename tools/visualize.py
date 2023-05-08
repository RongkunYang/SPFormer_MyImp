import argparse
import numpy as np
import os
import os.path as osp
import torch
from spformer.dataset import ScanNetDataset

nyu_id = ScanNetDataset.NYU_ID
text_label = ScanNetDataset.CLASSES
id2label = {}
for i in range(len(nyu_id)):
    id2label[str(nyu_id[i])] = text_label[i]

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255


def get_coords_color(opt):
    file = osp.join('data/scannetv2/val', opt.room_name + '_inst_nostuff.pth')
    xyz, rgb, superpoint, label, inst_label = torch.load(file)
    rgb = (rgb + 1) * 127.5
    instance_text_label = {}
    # same color order according to instance pointnum
    if opt.task == 'instance_gt':
        inst_label = inst_label.astype(int)
        print('Instance number: {}'.format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_rgb

    # same color order according to instance pointnum
    elif opt.task == 'instance_pred':
        instance_file = os.path.join(opt.prediction_path, 'pred_instance', opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result -{}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)

        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

        # sort score such that high score has a high priority for visualization
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]

        for i_ in range(len(masks) - 1, -1, -1):
            i = sort_inds[i_]
            mask_path = os.path.join(opt.prediction_path, 'pred_instance', masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if float(masks[i][2]) < 0.09:
                continue
            label_text = id2label[masks[i][1]]
            mask = np.array(open(mask_path).read().splitlines(), dtype=int)
            current_coords = xyz[mask==1,:]
            center_coord = current_coords.mean(axis=0)

            instance_text_label[label_text+'_'+str(i)+'_'+str(masks[i][-1])] = center_coord

            print('{} {}\t{}\t: pointnum: {}'.format(i, masks[i], label_text, mask.sum()))
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_pred_rgb

    elif opt.task == 'origin_pc':
        pass

    sem_valid = (label != -100)
    xyz = xyz[sem_valid]
    rgb = rgb[sem_valid]

    return xyz, rgb, instance_text_label


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []
    file = open(output_file, 'w')

    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                            int(color[1] * 255), int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', help='path to the prediction result', default='./result')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument('--task', help='instance_gt/instance_pred/origin_pc', default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format')
    opt = parser.parse_args()

    xyz, rgb, instance_text_label = get_coords_color(opt)
    if 0:
        instance_text_label = {}
    points = xyz[:, :3]
    colors = rgb / 255

    if opt.out:
        assert '.ply' in opt.out, 'output cloud file should be in FILE.ply'
        write_ply(points, colors, None, opt.out)
    else:
        import open3d as o3d
        import open3d.visualization.gui as gui

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer()
        # vis.show_settings = True
        # vis.create_window()
        vis.add_geometry('Points',pc)
        # vis.get_render_option().point_size = 1.5
        for label_name in instance_text_label.keys():
            vis.add_3d_label(instance_text_label[label_name], label_name)

        vis.show_skybox(False)
        vis.point_size=6
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

