# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Create plots with Matplotlib to visualise the result."""

import StringIO
import matplotlib.pyplot as plt
import numpy as np

HMR_JOINT_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head_top',
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
]

MSCOCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle'
]

coco_to_hmr = []
for name in MSCOCO_JOINT_NAMES:
  index = HMR_JOINT_NAMES.index(name)
  coco_to_hmr.append(index)

PARENTS_COCO_PLUS = [
    1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
]
COLOURS = []
for name in HMR_JOINT_NAMES:
  if name.startswith('left'):
    c = 'r'
  elif name.startswith('right'):
    c = 'g'
  else:
    c = 'm'
  COLOURS.append(c)


def plot_keypoints_2d(image,
                      joints_2d,
                      ax=None,
                      show_plot=False,
                      title='',
                      is_coco_format=False):
  """Plot 2d keypoints overlaid on image."""

  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  if hasattr(ax, 'set_axis_off'):
    ax.set_axis_off()

  if is_coco_format:
    kp = np.zeros((len(HMR_JOINT_NAMES), 2))
    kp[coco_to_hmr, :] = joints_2d
    joints_2d = kp

  if image is not None:
    ax.imshow(image)

  joint_colour = 'c' if not is_coco_format else 'b'
  s = 30 * np.ones(joints_2d.shape[0])
  for i in range(joints_2d.shape[0]):
    x, y = joints_2d[i, :]
    if x == 0 and y == 0:
      s[i] = 0

  ax.scatter(
      joints_2d[:, 0].squeeze(),
      joints_2d[:, 1].squeeze(),
      s=30,
      c=joint_colour)

  for idx_i, idx_j in enumerate(PARENTS_COCO_PLUS):
    if idx_j >= 0:
      pair = [idx_i, idx_j]
      x, y = joints_2d[pair, 0], joints_2d[pair, 1]
      if x[0] > 0 and y[0] > 0 and x[1] > 0 and y[1] > 0:
        ax.plot(x.squeeze(), y.squeeze(), c=COLOURS[idx_i], linewidth=1.5)

  ax.set_xlim([0, image.shape[1]])
  ax.set_ylim([image.shape[0], 0])

  if title:
    ax.set_title(title)

  if show_plot:
    plt.show()

  return ax


def plot_summary_figure(img,
                        joints_2d,
                        rend_img_overlay,
                        rend_img,
                        rend_img_vp1,
                        rend_img_vp2,
                        save_name=None):
  """Create plot to visulise results."""

  fig = plt.figure(1, figsize=(20, 12))
  plt.clf()

  plt.subplot(231)
  plt.imshow(img)
  plt.title('Input')
  plt.axis('off')

  ax_skel = plt.subplot(232)
  ax_skel = plot_keypoints_2d(img, joints_2d, ax_skel)
  plt.title('Joint Projection')
  plt.axis('off')

  plt.subplot(233)
  plt.imshow(rend_img_overlay)
  plt.title('3D Mesh overlay')
  plt.axis('off')

  plt.subplot(234)
  plt.imshow(rend_img)
  plt.title('3D mesh')
  plt.axis('off')

  plt.subplot(235)
  plt.imshow(rend_img_vp1)
  plt.title('Other viewpoint (+60 degrees)')

  plt.axis('off')
  plt.subplot(236)
  plt.imshow(rend_img_vp2)
  plt.title('Other viewpoint (-60 degrees)')
  plt.axis('off')

  plt.draw()

  if save_name is not None:
    buf = StringIO.StringIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)

    with open(save_name, 'w') as fp:
      fp.write(buf.read(-1))
  else:
    plt.show()

  return fig

