# [**[CVPR-2024] A Dual Augmentor Framework for Domain Generalization in 3D Human Pose Estimation**](https://arxiv.org/abs/2403.11310)

### Prerequisites:
- Datasets: Please follow [**PoseAug**](https://github.com/jfzhang95/PoseAug) and [**AdaptPose**](https://github.com/mgholamikn/AdaptPose).
- Environments: Please follow [**PoseAug**](https://github.com/jfzhang95/PoseAug).
- Backbone: Here we only provide the [**VideoPose3D**](https://dariopavllo.github.io/VideoPose3D/) as the 2D-lifting-3D backbone. You can try other backbones by adding new directories in "model_baseline"
- Pretraining and Evaluation: We do not contain these parts in the repo. You can either follow previous works like [**PoseAug**](https://github.com/jfzhang95/PoseAug) and [**AdaptPose**](https://github.com/mgholamikn/AdaptPose) to implement or write it by yourself.

### Run Training Codes:
```
python3 run_daf_dg.py --note poseaug --posenet_name 'videopose' --checkpoint './checkpoint' --keypoints gt
```

### Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{peng2024dual,
  title={A Dual-Augmentor Framework for Domain Generalization in 3D Human Pose Estimation},
  author={Peng, Qucheng and Zheng, Ce and Chen, Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2240--2249},
  year={2024}
}
```
### Acknowledge

Borrow a lot from [**PoseAug**](https://github.com/jfzhang95/PoseAug).

