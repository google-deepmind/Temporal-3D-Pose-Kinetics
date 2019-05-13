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

"""Visualise bundle adjustment results.

Example to run:
python run_visualise.py --filename KV4jIAq3WJo_155_165.pkl
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import errno
import os
import subprocess
import sys

from absl import flags
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plot_utils
import skvideo.io
from third_party.activity_net.download import download_clip
import third_party.hmr.renderer as vis_util

# Input
flags.DEFINE_string('filename', '', 'The annoation pickle file')
flags.DEFINE_string('smpl_face_path', 'smpl_faces.npy',
                    'Path to smpl model face file.')

# Output
flags.DEFINE_string(
    'output_dir', 'results', 'Where to write results to.'
    'Directory automatically created.')


def mkdir(dirname):
  """Create directory if it does not exist."""
  try:
    os.makedirs(dirname)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
      raise


def im_save_cv(image, filename):
  """Write image with OpenCV, converting from BGR to RGB format."""
  cv2.imwrite(filename, image[:, :, (2, 1, 0)])


def visualize(img,
              joints,
              vertices,
              camera,
              image_name,
              output_dir,
              renderer=None,
              color_id=0):
  """Renders the result in original image coordinate frame.

    Args:
        img: The image
        joints: 2D keypoints, in the image coordinate frame.
        vertices: Vertices of the SMPL mesh.
        camera: Camera predicted.
        image_name: Name of image for saving.
        output_dir: Directory to save results to
        renderer: Renderer object to use.
        color_id: 0 is blue, and 1 is light pink. For the visualisation. The
          colours are defined in the renderer.
  """

  cam_for_render = camera * img.shape[0]
  vert_shifted = np.copy(vertices)

  # Approximate an orthographic camera:
  # move points away and adjust the focal length to zoom in.
  vert_shifted[:, -1] = vert_shifted[:, -1] + 100.
  cam_for_render[0] *= 100.

  rend_img_overlay = renderer(
      vert_shifted,
      cam=cam_for_render,
      img=img,
      do_alpha=True,
      color_id=color_id)
  rend_img = renderer(
      vert_shifted,
      cam=cam_for_render,
      img_size=img.shape[:2],
      color_id=color_id)
  rend_img_vp1 = renderer.rotated(
      vert_shifted,
      60,
      cam=cam_for_render,
      img_size=img.shape[:2],
      color_id=color_id)
  rend_img_vp2 = renderer.rotated(
      vert_shifted,
      -60,
      cam=cam_for_render,
      img_size=img.shape[:2],
      color_id=color_id)

  save_name = os.path.join(output_dir, image_name + '.jpg')
  fig = plot_utils.plot_summary_figure(img, joints, rend_img_overlay, rend_img,
                                       rend_img_vp1, rend_img_vp2, save_name)
  plt.close(fig)


def transform_keypoints_to_image(keypoints, img):
  """Transform keypoints from range [0, 1] to image coordinates."""

  keypoints[:, :, 0] *= img.shape[0]
  keypoints[:, :, 1] *= img.shape[
      0]  # The saved keypoints are scaled by image height.

  return keypoints


def parse_filename(filename):
  """Parse filename of the pickle file."""

  name = os.path.basename(filename)
  name = name.replace('.pkl', '')
  tokens = name.split('_')
  end_time = int(tokens[-1])
  start_time = int(tokens[-2])
  video_id = '_'.join(tokens[0:-2])

  return video_id, start_time, end_time


def get_frame_rate(video_path):
  """Get frame rate of the video from its metadata."""

  meta_data = skvideo.io.ffprobe(video_path)
  if 'video' in meta_data.keys():
    meta_data = meta_data['video']

  if '@avg_frame_rate' in meta_data:
    frame_rate = eval(meta_data['@avg_frame_rate'])
  else:
    frame_rate = None

  return frame_rate


def video_from_images(directory, save_name):
  """Create video from images saved in directory using ffmpeg."""

  command = [
      'ffmpeg', '-framerate', '25', '-pattern_type',
      'glob -i \'{}/*.jpg\''.format(directory), '-c:v', 'libx264', '-pix_fmt',
      'yuv420p', '-loglevel', 'panic', save_name
  ]
  command = ' '.join(command)
  try:
    _ = subprocess.check_output(
        command, shell=True, stderr=subprocess.STDOUT)
  except:
    pass


def load_pickle(filename):
  """Read pickle file."""

  with open(filename) as fp:
    data = pickle.load(fp)
  return data


def main(config):

  data = load_pickle(config.filename)

  video_id, start_time, end_time = parse_filename(config.filename)
  video_path = '/tmp/' + video_id + '.mp4'

  status, message = download_clip(video_id, video_path, start_time, end_time)

  if not status:
    print('Video not downloaded')
    print(message)
    sys.exit()

  video = skvideo.io.vread(video_path)
  frame_rate = get_frame_rate(video_path)

  if not frame_rate:
    print('Error. Could not determine frame rate of video')
    sys.exit()

  output_dir = os.path.join(config.output_dir, video_id)
  mkdir(output_dir)

  keypoints = transform_keypoints_to_image(data['2d_keypoints'],
                                           video[0].squeeze())
  renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

  for i in range(data['time'].size):

    idx = int(round(data['time'][i] * frame_rate))
    if idx >= video.shape[0]:
      break

    img = video[idx].squeeze()
    image_name = '{:>04}'.format(i)

    visualize(
        img,
        joints=keypoints[i].squeeze(),
        vertices=data['vertices'][i].squeeze(),
        camera=data['camera'][i].squeeze(),
        image_name=image_name,
        output_dir=output_dir,
        renderer=renderer)

    if i % 20 == 0:
      print('Processed {:3d} / {:3d}'.format(i + 1, data['time'].size))

  video_from_images(output_dir, os.path.join(output_dir, video_id + '.mp4'))


if __name__ == '__main__':
  config_ = flags.FLAGS
  config_(sys.argv)

  main(config_)
