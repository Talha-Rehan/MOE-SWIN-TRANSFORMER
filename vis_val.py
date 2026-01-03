from mmdet.apis import DetInferencer

CFG = 'configs/fire/faster-rcnn_swin-t_fpn_fire.py'
CKPT = 'work_dirs/faster-rcnn_swin-t_fpn_fire/best_coco_bbox_mAP_epoch_10.pth'
VAL_DIR = 'data/images/val'
OUT_DIR = 'work_dirs/val_vis'
THR = 0.30

if __name__ == '__main__':
    inferencer = DetInferencer(CFG, CKPT, device='cuda:0')
    inferencer(
        inputs=VAL_DIR,
        out_dir=OUT_DIR,
        pred_score_thr=THR,
        no_save_vis=False,
        no_save_pred=True,
        print_result=False
    )
