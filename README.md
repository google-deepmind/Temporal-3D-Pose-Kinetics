Exploiting temporal context for 3D human pose estimation in the wild
==

[Exploiting temporal context for 3D human pose estimation in the wild](http://arxiv.org/abs/1905.04266) uses temporal information from videos to correct errors in single-image 3D pose estimation.  In this repository, we provide results from applying this algorithm on the [Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset.  Note that this is not an exhaustive labeling: at most one person is labeled per frame, and frames which the algorithm has identified as outliers are not labeled.

The archive contains a single `.pkl` file for each video where bundle adjustment succeeded.  Let `N` be the number of frames that the algorithm considers inliers.  Then the `.pkl` file contains a map with the following keys:

* `time`: Array of size `N`, where each element is the time in seconds since the start of the 10-second kinetics clip (not the start of the whole video)
* `smpl_shape`: Array of size `Nx10`, where each row is the SMPL shape for one example.
* `smpl_pose`: Array of size `Nx72`, where each row is the SMPL pose for one example.
* `3d_keypoints`: Array of size `Nx24x3` where each slice is the 19 cocoplus joints obtained from the SMPL model using the custom keypoint regressor described below.
* `2d_keypoints`: Array of size `Nx19x2`, where each slice is the 19 cocoplus joints reprojected from the SMPL model, using the custom keypoint regressor described below, in `(x,y)` coordinates.  These coordinates are normalized to the image frame: therefore, (0, 0) and (1,1) are the top-left and bottom-right corners respectively.
* `cameras`: Array of size `Nx3`, containing the translation and scale that maps the SMPL 3D joint locations to `2d_keypoints`.  `cameras[:,0]` is scale and `cameras[:,1:3]` is translation.  Thus, if `x` is a `19x3` array of 3D keypoints in the format `(x,y,z)` produced byt the SMPL model, then `2d_keypoints` can be computed as `cameras[:,0:1]*(x[:,0:2]+cameras[:,1:3])`.
* `vertices`: Array of size `Nx6890x3`. These are the vertices of the SMPL mesh computed from `smpl_shape` and `smpl_pose` computing with the neutral body model from [HMR](https://github.com/akanazawa/hmr).

The dataset can be downloaded [here](https://storage.cloud.google.com/temporal-3d-pose-kinetics/temporal_3d_pose_kinetics.tar.gz) (325 GB), as well as an significantly smaller archive which does not contain `vertices`, but is otherwise identical, [here](https://storage.cloud.google.com/temporal-3d-pose-kinetics/temporal_3d_pose_kinetics_noverts.tar.gz) (2.7 GB).

## Joint regressor

We also have a custom [joint regressor](https://storage.cloud.google.com/temporal-3d-pose-kinetics/custom_joint_regressor.pkl) that is specific to our pose estimator (since there are slight differences between the 2D joints we used for bundle adjustment and those used for SMPL).  This is a `6890x19` array that can be used as a drop-in replacement for the `cocoplus_regressor` that is distributed in the public [HMR repository](https://github.com/akanazawa/hmr), and is required to extract the `3d_keypoints` above from the estimated poses.  It was learned using ground-truth from the [Human3.6m dataset](http://vision.imar.ro/human3.6m/).

## Pretrained Model
This [Tensorflow checkpoint](https://storage.cloud.google.com/temporal-3d-pose-kinetics/model-894621.tar.gz) was trained using the procedure outlined in our paper.  That is, it uses the above dataset as well as standard HMR 3D data.  The checkpoint is compatible with [HMR](https://github.com/akanazawa/hmr).

## Visualising data

- You need to install [`youtube-dl`](https://github.com/ytdl-org/youtube-dl) and [`ffmpeg`](http://ffmpeg.org) to download the Kinetics videos to visualise.
- Download the faces of the SMPL mesh for visualisation: `wget https://github.com/akanazawa/hmr/raw/master/src/tf_smpl/smpl_faces.npy`
- Download the Kinetics download script from [ActivityNet](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py) and place it in `third_party/activity_net`. This can be done with: `wget https://raw.githubusercontent.com/activitynet/ActivityNet/master/Crawler/Kinetics/download.py -P third_party/activity_net`. We tested with commit 530ac3a of the download script.
- The python packages needed are in `requirements.txt`. We recommend creating a new virtual environment, and running `pip install -r requirements.txt`.

To run the demo:

`python run_visualise --filename <path_to_downloaded_pickle_file>`

## Credits
- The renderer to visualise the SMPL model is from [HMR]( https://github.com/akanazawa/hmr)
- The Kinetics download script is from [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)


## Reference
If you use this data, please cite

```tex
@InProceedings{Arnab_CVPR_2019,
    author = {Arnab, Anurag* and 
              Doersch, Carl* and 
              Zisserman, Andrew},
    title = {Exploiting temporal context for 3D human pose estimation in the wild},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```
