import numpy as np 

d = "/home/ubuntu/3drecon/gaussian-splatting/data/demo/poses.npz"
x = np.load(d)
"""
['near', 'far', 'poses', 'intrinsics', 'pose_type', 'img_paths', 'points_xyz', 'points_rgb']

poses: (N,4,4)
intrinsics" (4,4)
"""

