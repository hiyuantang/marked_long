import time
import sys
import numpy as np
import os
import snakes2

class Args:
    def __init__(self,
                 contour_path = './contour', batch_id=0, n_images = 200000,
                 window_size=[256,256], padding=22, antialias_scale = 4,
                 LABEL =1, seed_distance= 27, marker_radius = 3,
                 contour_length=15, distractor_length=5, num_distractor_snakes=6, snake_contrast_list=[1.], use_single_paddles=True,
                 max_target_contour_retrial = 4, max_distractor_contour_retrial = 4, max_paddle_retrial=2,
                 continuity = 1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast_list = snake_contrast_list
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list # if multiple elements in a list, a number will be sampled in each IMAGE
        self.paddle_contrast_list = paddle_contrast_list # if multiple elements in a list, a number will be sampled in each PADDLE

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
current_id = int(sys.argv[2])
args.batch_id = current_id
total_images = int(sys.argv[3])
args.n_images = total_images/num_machines
dataset_root = '/Users/junkyungkim/Desktop/pathfinder/' #'/media/data_cifs/pathfinder_seg/'

if len(sys.argv)==4:
    print('Using default path...')
elif len(sys.argv)==5:
    print('Using custom save path...')
    dataset_root = str(sys.argv[4])

args.antialias_scale = 4
args.paddle_margin_list = [3]

args.window_size = [300,300]
args.marker_radius = 5 #3
args.contour_length = 6 # from 6 to 14, with steps of 50%
args.paddle_thickness = 2 #1.5
args.antialias_scale = 2
args.continuity = 1.8  # from 1.8 to 0.8, with steps of 66%
args.distractor_length = args.contour_length / 3
args.num_distractor_snakes = 30 / args.distractor_length
args.snake_contrast_list = [1.0]
args.use_single_paddles = False
args.segmentation_task = False # False
args.segmentation_task_double_circle = False

################################# DS: BASELINE
dataset_subpath = 'curv_baseline'
args.contour_path = os.path.join(dataset_root, dataset_subpath)
snakes2.from_wrapper(args)

################################# DS: snake length
for cl in [18, 21]: #[9, 14]:
    args.contour_length = cl
    args.distractor_length = cl/3
    dataset_subpath = 'curv_contour_length_' + str(cl)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    snakes2.from_wrapper(args)
args.contour_length = 6
args.distractor_length = 2

################################# DS: snake inter-paddle continuity
for ct in [1.8, 0.8]:
    args.continuity = ct
    dataset_subpath = 'curv_continuity_' + str(ct)
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    # snakes2.from_wrapper(args)
args.continuity = 1.8

################################# DS: (REST OF THE 2-way MATRIX)
# NOT IMPLEMENTED YET
for cl in [6, 12, 15, 18]:
    for ct in [2.7, 1.2, 0.8, 0.6]:
        if (cl==6) & (ct ==2.7):
            args.continuity = ct
            args.contour_length = cl
            args.distractor_length = cl / 3
            # POS
            dataset_subpath = 'curv_continuity_' + str(ct) + '_length_' + str(cl)
            args.contour_path = os.path.join(dataset_root, dataset_subpath)
            # snakes2.from_wrapper(args)
        else:
            print('not implemented.')


elapsed = time.time() - t
print('n_totl_imgs (per condition) : ', str(total_images))
print('ELAPSED TIME OVER ALL CONDITIONS : ', str(elapsed))
