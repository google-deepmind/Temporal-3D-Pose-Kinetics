"""
https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import subprocess
import uuid


def download_clip(video_identifier,
                  output_filename,
                  start_time,
                  end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
  """Download a video from youtube if exists and is not blocked.


    Args:
        video_identifier: str. Unique YouTube video identifier (11 characters)
        output_filename: str File path where the video will be stored.
        start_time: float Indicates the beginning time in seconds from where the
          video will be trimmed.
        end_time: float Indicates the ending time in seconds of the trimmed
          video.

    Returns:
        status: boolean. Whether the downloaded succeeded
        message: str. Error message if download did not succeed

  """
  # Defensive argument checking.
  assert isinstance(video_identifier, str), 'video_identifier must be string'
  assert isinstance(output_filename, str), 'output_filename must be string'
  assert len(video_identifier) == 11, 'video_identifier must have length 11'

  if os.path.exists(output_filename):
    return True, 'Downloaded'

  status = False
  # Construct command line for getting the direct video link.
  tmp_filename = os.path.join(tmp_dir, '%s.%%(ext)s' % uuid.uuid4())
  command = [
      'youtube-dl', '--quiet', '--no-warnings', '-f', 'mp4', '-o',
      '"%s"' % tmp_filename,
      '"%s"' % (url_base + video_identifier)
  ]
  command = ' '.join(command)
  attempts = 0
  while True:
    try:
      _ = subprocess.check_output(
          command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
      attempts += 1
      if attempts == num_attempts:
        return status, err.output
    else:
      break

  tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
  # Construct command to trim the videos (ffmpeg required).
  command = [
      'ffmpeg', '-i',
      '"%s"' % tmp_filename, '-ss',
      str(start_time), '-t',
      str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy', '-threads',
      '1', '-loglevel', 'panic',
      '"%s"' % output_filename
  ]
  command = ' '.join(command)
  try:
    output = subprocess.check_output(
        command, shell=True, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as err:
    return status, err.output

  # Check if the video was successfully saved.
  status = os.path.exists(output_filename)
  os.remove(tmp_filename)
  return status, 'Downloaded'

