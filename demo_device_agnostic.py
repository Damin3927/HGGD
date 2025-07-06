# filepath: /Users/damin/Programs/HGGD/demo_device_agnostic.py
"""
HGGD Demo Script - Hierarchical Grasp Generation and Detection

This script demonstrates the grasp detection pipeline using the HGGD framework.
It takes RGB and depth images as input and outputs 6D grasp poses.
The pipeline consists of two main networks:
1. AnchorNet: Detects 2D grasps from RGB-D images
2. LocalGraspNet: Refines 2D grasps to 6D poses using point cloud data

Usage:
    python demo.py --checkpoint-path PATH_TO_CHECKPOINT --rgb-path PATH_TO_RGB 
                  --depth-path PATH_TO_DEPTH --input-h HEIGHT --input-w WIDTH
                  --anchor-num NUM --all-points-num NUM --center-num NUM --group-num NUM
                  --device [cuda|cpu]
"""

import argparse
import os
import random
from time import time

import numpy as np
# import open3d as o3d  # For 3D visualization
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from hggd.dataset.config import get_camera_intrinsic
from hggd.dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from hggd.dataset.pc_dataset_tools import data_process, feature_fusion
from hggd.models.anchornet import AnchorGraspNet
from hggd.models.localgraspnet import PointMultiGraspNet
from train_utils import *  # Import utilities for training/logging

parser = argparse.ArgumentParser(description='HGGD Demo Script for detecting 6D grasps from RGB-D images')

# Main path parameters
parser.add_argument('--checkpoint-path', default=None, 
                    help='Path to the model checkpoint file')

# Input image parameters
parser.add_argument('--rgb-path', help='Path to the RGB image')
parser.add_argument('--depth-path', help='Path to the depth image')

# 2D grasp detection parameters
parser.add_argument('--input-h', type=int, 
                    help='Height for network input')
parser.add_argument('--input-w', type=int, 
                    help='Width for network input')
parser.add_argument('--sigma', type=int, default=10,
                    help='Gaussian sigma for heatmap generation')
parser.add_argument('--use-depth', type=int, default=1,
                    help='Whether to use depth image (1 for yes, 0 for no)')
parser.add_argument('--use-rgb', type=int, default=1,
                    help='Whether to use RGB image (1 for yes, 0 for no)')
parser.add_argument('--ratio', type=int, default=8,
                    help='Downsampling ratio for feature maps')
parser.add_argument('--anchor-k', type=int, default=6,
                    help='Number of anchor rotations')
parser.add_argument('--anchor-w', type=float, default=50.0,
                    help='Default anchor width in pixels')
parser.add_argument('--anchor-z', type=float, default=20.0,
                    help='Default anchor depth in mm')
parser.add_argument('--grid-size', type=int, default=8,
                    help='Grid size for grasp sampling and NMS')

# Point cloud processing parameters
parser.add_argument('--anchor-num', type=int,
                    help='Number of anchors for rotation and approach')
parser.add_argument('--all-points-num', type=int,
                    help='Maximum number of points to sample from point cloud')
parser.add_argument('--center-num', type=int,
                    help='Number of grasp centers to consider')
parser.add_argument('--group-num', type=int,
                    help='Number of points in each local point group')

# Grasp detection thresholds and parameters
parser.add_argument('--heatmap-thres', type=float, default=0.01,
                    help='Threshold for heatmap values')
parser.add_argument('--local-k', type=int, default=10,
                    help='Number of local neighbors for point cloud processing')
parser.add_argument('--local-thres', type=float, default=0.01,
                    help='Threshold for local grasp detection')
parser.add_argument('--rotation-num', type=int, default=1,
                    help='Number of rotation augmentations')

# Miscellaneous parameters
parser.add_argument('--random-seed', type=int, default=123, 
                    help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                    help='Device to run inference on (cuda or cpu)')

args = parser.parse_args()


class PointCloudHelper:
    """
    Helper class for point cloud processing from RGB-D images.
    
    This class handles the conversion between depth images and 3D point clouds,
    including operations like backprojection, sampling, and feature extraction.
    It manages camera intrinsics and coordinate transformations necessary for
    accurate 3D reconstruction from 2D images.
    """

    def __init__(self, all_points_num: int, device: torch.device) -> None:
        """
        Initialize the PointCloudHelper with camera parameters.
        
        Args:
            all_points_num (int): Maximum number of points to sample from the point cloud
            device (torch.device): Device to run computations on (CPU or CUDA)
        """
        # Set maximum number of points to sample
        self.all_points_num = all_points_num
        self.device = device
        
        # Define downsampled output shape for feature maps
        self.output_shape = (80, 45)
        
        # Load camera intrinsic parameters (focal length, principal point)
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Focal lengths
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # Principal point
        
        # Create coordinate maps for original resolution (1280x720)
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        # Convert pixel coordinates to normalized camera coordinates
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float().to(device)
        self.points_y = torch.from_numpy(points_y).float().to(device)
        
        # Create coordinate maps for downsampled resolution (output_shape)
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                np.arange(self.output_shape[0]))
        factor = 1280 / self.output_shape[0]  # Scale factor for intrinsics
        # Convert downsampled pixel coordinates to normalized camera coordinates
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float().to(device)
        self.points_y_downscale = torch.from_numpy(points_y).float().to(device)

    def to_scene_points(self,
                        rgbs: torch.Tensor,
                        depths: torch.Tensor,
                        include_rgb=True):
        """
        Convert RGB-D images to 3D point clouds with color information.
        
        This method backprojects 2D depth images into 3D space using the camera intrinsics,
        and optionally adds RGB color information to each point.
        
        Args:
            rgbs (torch.Tensor): Batch of RGB images [B, 3, H, W]
            depths (torch.Tensor): Batch of depth images [B, H, W]
            include_rgb (bool): Whether to include RGB color information
            
        Returns:
            tuple:
                - points_all (torch.Tensor): Batch of point clouds [B, all_points_num, 3+3*include_rgb]
                - idxs (list): Indices of sampled points
                - masks (torch.Tensor): Masks indicating valid depth values
        """
        batch_size = rgbs.shape[0]
        # Number of features per point: XYZ + optional RGB
        feature_len = 3 + 3 * include_rgb
        # Initialize point cloud tensor with placeholder values (-1)
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).to(self.device)
        
        # Calculate 3D coordinates from depth values
        idxs = []
        # Create mask for valid depth values (depth > 0)
        masks = (depths > 0)
        # Convert depth from mm to meters
        cur_zs = depths / 1000.0  
        # Compute X and Y coordinates using normalized coordinates and depth
        cur_xs = self.points_x * cur_zs
        cur_ys = self.points_y * cur_zs
        
        for i in range(batch_size):
            # Stack XYZ coordinates for this batch item
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], dim=-1)
            # Filter out points with invalid depth
            mask = masks[i]
            points = points[mask]
            # Get RGB values for valid points
            colors = rgbs[i][:, mask].T

            # Randomly sample points if we have more than required
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)),
                                         self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # Save indices for feature fusion later
                idxs.append(cur_idxs)

            # Combine geometric (XYZ) and appearance (RGB) features
            if include_rgb:
                # Use dim instead of axis for torch tensors
                points_all[i, :points.shape[0]] = torch.cat([points, colors], dim=1)
            else:
                points_all[i, :points.shape[0]] = points
                
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        """
        Convert depth images to feature maps of XYZ coordinates.
        
        This method projects depth images into 3D space and creates feature maps where
        each pixel contains the corresponding 3D coordinate. These maps are used for
        point cloud feature extraction in later stages.
        
        Args:
            depths (torch.Tensor): Batch of depth images [B, H, W]
            
        Returns:
            torch.Tensor: XYZ feature maps [B, 3, output_height, output_width]
        """
        # Downsample the depth image to the output shape
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).to(self.device)
        
        # Convert depth from mm to meters
        cur_zs = downsample_depths / 1000.0
        
        # Calculate X and Y coordinates using downsampled normalized coordinates and depth
        cur_xs = self.points_x_downscale * cur_zs
        cur_ys = self.points_y_downscale * cur_zs
        
        # Stack coordinates to create XYZ feature maps (use dim instead of axis)
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], dim=-1)
        
        # Rearrange dimensions to [B, 3, H, W] format
        return xyzs.permute(0, 3, 1, 2)


def inference(view_points,
              xyzs,
              x,
              ori_depth,
              anchornet,
              localnet,
              anchors,
              device,
              vis_heatmap=False,
              vis_grasp=True):
    """
    Perform the complete grasp detection pipeline.
    
    This function runs the hierarchical grasp detection pipeline:
    1. First detects 2D grasps using AnchorNet
    2. Then refines them to 6D grasps using LocalGraspNet
    3. Finally performs collision checking and non-maximum suppression
    
    Args:
        view_points (torch.Tensor): Point cloud with XYZ coordinates and RGB values
        xyzs (torch.Tensor): XYZ feature maps
        x (torch.Tensor): Input tensor combining RGB and depth [B, 4, H, W]
        ori_depth (torch.Tensor): Original depth image
        anchornet: The network for 2D grasp detection
        localnet: The network for 3D grasp refinement
        anchors: Dictionary containing anchor points for rotation and approach
        device (torch.device): Device to run inference on (CPU or CUDA)
        vis_heatmap (bool, optional): Whether to visualize heatmaps. Defaults to False.
        vis_grasp (bool, optional): Whether to visualize detected grasps. Defaults to True.
        
    Returns:
        pred_gg: Predicted grasp group with filtered 6D grasps
    """
    with torch.no_grad():
        # Step 1: 2D grasp detection with AnchorNet
        # AnchorNet outputs: 2D predictions and per-point features
        pred_2d, perpoint_features = anchornet(x)

        # Process outputs to get location map, classification mask, and grasp parameter offsets
        loc_map, cls_mask, theta_offset, height_offset, width_offset = \
            anchor_output_process(*pred_2d, sigma=args.sigma)

        # Detect 2D grasp rectangles with parameters (x, y, theta, width, height)
        rect_gg = detect_2d_grasp(loc_map,
                                  cls_mask,
                                  theta_offset,
                                  height_offset,
                                  width_offset,
                                  ratio=args.ratio,
                                  anchor_k=args.anchor_k,
                                  anchor_w=args.anchor_w,
                                  anchor_z=args.anchor_z,
                                  mask_thre=args.heatmap_thres,
                                  center_num=args.center_num,
                                  grid_size=args.grid_size,
                                  grasp_nms=args.grid_size,
                                  reduce='max')

        # Check if any grasps were detected
        if rect_gg.size == 0:
            print('No 2d grasp found')
            return None

        # Visualize intermediate results if requested
        if vis_heatmap:
            # Extract RGB and depth for visualization (move to CPU for visualization)
            rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
            resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
            resized_rgb = np.array(
                resized_rgb.resize((args.input_w, args.input_h))) / 255.0
            depth_t = ori_depth.cpu().numpy().squeeze().T
            
            # Create visualization with input RGB, depth, grasp heatmap, and 2D grasp predictions
            plt.subplot(221)
            plt.imshow(rgb_t)
            plt.subplot(222)
            plt.imshow(depth_t)
            plt.subplot(223)
            plt.imshow(loc_map.squeeze().cpu().numpy().T, cmap='jet')  # Grasp quality heatmap
            plt.subplot(224)
            rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0)  # Draw 2D grasps
            plt.imshow(rect_rgb)
            plt.tight_layout()
            plt.show()

        # Step 2: Feature fusion between 2D features and 3D point cloud
        # Combine per-point features from image with XYZ coordinates
        points_all = feature_fusion(view_points[..., :3], perpoint_features,
                                    xyzs)
        
        # Prepare data for the local refinement network
        rect_ggs = [rect_gg]
        # Extract point groups centered at each grasp location
        pc_group, valid_local_centers = data_process(
            points_all,
            ori_depth,
            rect_ggs,
            args.center_num,
            args.group_num, (args.input_w, args.input_h),
            min_points=32,
            is_training=False)
        rect_gg = rect_ggs[0]
        # Remove batch dimension since we're processing one image
        points_all = points_all.squeeze()

        # Step 3: Prepare grasp parameters for the LocalGraspNet
        # Extract rotation (theta), width, and depth for each 2D grasp
        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]  # Grasp rotation angles
        g_ws = rect_gg.widths[None]      # Grasp widths
        g_ds = rect_gg.depths[None]      # Grasp depths
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device=device)

        # Step 4: Run LocalGraspNet to refine 2D grasps to 6D
        # LocalGraspNet outputs: classification scores, grasp orientation predictions, and position offsets
        _, pred, offset = localnet(pc_group, grasp_info)

        # Step 5: Convert network predictions to 6D grasps
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg,
                                                pred,
                                                offset,
                                                valid_local_centers,
                                                (args.input_w, args.input_h),
                                                anchors,
                                                k=args.local_k)

        # Step 6: Generate 6D grasp poses and perform collision checking
        # Convert rectangles to 6D grasp representations
        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        # Filter grasps by collision detection with point cloud
        pred_gg, _ = collision_detect(points_all,
                                      pred_grasp_from_rect,
                                      mode='graspnet')

        # Step 7: Apply non-maximum suppression to remove redundant grasps
        pred_gg = pred_gg.nms()

        # Step 8: Visualize final grasp results if requested
        if vis_grasp:
            print('pred grasp num ==', len(pred_gg))
            # Convert grasps to Open3D geometries for visualization
            grasp_geo = pred_gg.to_open3d_geometry_list()
            # Extract point cloud data for visualization
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            # Create Open3D point cloud
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            # Show point cloud with grasp poses
            o3d.visualization.draw_geometries([vispc] + grasp_geo)
            
        return pred_gg


if __name__ == '__main__':
    """
    Main execution of the HGGD demo.
    
    This section handles:
    1. Setting up the point cloud processing helper
    2. Configuring GPU settings and random seeds
    3. Initializing and loading the grasp detection models
    4. Processing input RGB-D images
    5. Running the grasp detection pipeline
    6. Measuring inference performance
    """
    # Configure numpy and PyTorch numeric display settings
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    
    # Set up device (CPU or CUDA)
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    
    # Configure device specific settings
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False  # Disabled for deterministic behavior
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for computation. This will be slower than using GPU.")
    
    # Set random seeds for reproducibility across all random number generators
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set up point cloud transform helper with configured maximum number of points
    pc_helper = PointCloudHelper(all_points_num=args.all_points_num, device=device)

    # Initialize the two-stage model architecture:
    # 1. AnchorNet: Detects initial 2D grasps from RGB-D input
    #    - in_dim=4: RGB (3 channels) + depth (1 channel)
    anchornet = AnchorGraspNet(in_dim=4,
                              ratio=args.ratio,
                              anchor_k=args.anchor_k).to(device)
                              
    # 2. LocalGraspNet: Refines 2D grasps to 6D poses using point cloud
    #    - info_size=3: theta, width, depth of initial 2D grasp
    #    - k_cls=args.anchor_num**2: total anchor classes (combinations of approach and rotation)
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2).to(device)

    # Load pre-trained model weights from checkpoint
    if device.type == 'cuda':
        check_point = torch.load(args.checkpoint_path)
    else:
        # Load on CPU if CUDA is not available
        check_point = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])
    
    # Set up anchor points for approach and rotation discretization
    # These anchors define the discrete rotation and approach angles for the 6D grasp poses
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).to(device)
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2  # Center of each bin
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}  # Initial uniform anchors
    # Override with learned anchors from checkpoint (better than uniform distribution)
    anchors['gamma'] = check_point['gamma'].to(device)  # Approach angle anchors
    anchors['beta'] = check_point['beta'].to(device)    # Rotation angle anchors
    logging.info('Using saved anchors')
    print('-> loaded checkpoint %s ' % (args.checkpoint_path))

    # Set networks to evaluation mode (disables dropout, etc.)
    anchornet.eval()
    localnet.eval()

    # Load and preprocess input RGB and depth images
    # Load depth image and convert to numpy array (values in mm)
    ori_depth = np.array(Image.open(args.depth_path))
    # Load RGB image and normalize to [0,1] range
    ori_rgb = np.array(Image.open(args.rgb_path)) / 255.0
    
    # Clip depth values to valid range (0-1000mm or 0-1m)
    # This removes invalid readings that may exist in the depth image
    ori_depth = np.clip(ori_depth, 0, 1000)
    
    # Convert numpy arrays to PyTorch tensors with appropriate dimensions
    # RGB: [H,W,3] -> [1,3,W,H] (adding batch dimension, moving channels to PyTorch format)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device=device, dtype=torch.float32)
    
    # Depth: [H,W] -> [1,W,H] (adding batch dimension, transposing for PyTorch format)
    ori_depth = torch.from_numpy(ori_depth).T[None]
    ori_depth = ori_depth.to(device=device, dtype=torch.float32)

    # Convert RGB-D data to 3D point cloud representation
    # This generates the full point cloud with color information
    view_points, _, _ = pc_helper.to_scene_points(ori_rgb,
                                                  ori_depth,
                                                  include_rgb=True)
    
    # Generate XYZ coordinate maps for feature extraction
    # These maps encode 3D position information at each pixel
    xyzs = pc_helper.to_xyz_maps(ori_depth)

    # Preprocess images to network input resolution and format
    # Resize RGB image to network input dimensions
    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    
    # Resize depth image to network input dimensions
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    # Convert depth from mm to meters and normalize to [-1,1] range
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    
    # Concatenate depth and RGB channels to form network input (RGBD)
    # Format: [batch_size, channels, height, width] = [1, 4, input_h, input_w]
    # Channel order: [depth, R, G, B]
    x = torch.concat([depth[None], rgb], 1)
    x = x.to(device=device, dtype=torch.float32)

    # Run the full grasp detection pipeline once with visualization enabled
    # This first run will display the intermediate results and final grasp poses
    print("Running inference with visualization...")
    pred_gg = inference(view_points,
                        xyzs,
                        x,
                        ori_depth,
                        anchornet,
                        localnet,
                        anchors,
                        device,
                        vis_heatmap=True,  # Show heatmap visualization
                        vis_grasp=True)    # Show 3D grasp visualization

    # Performance benchmarking: measure average inference time over multiple runs
    print("Running performance benchmark...")
    start = time()
    T = 10 if device.type == 'cpu' else 100  # Fewer iterations for CPU (slower)
    for _ in range(T):
        # Run inference without visualization for timing purposes
        pred_gg = inference(view_points,
                            xyzs,
                            x,
                            ori_depth,
                            anchornet,
                            localnet,
                            anchors,
                            device,
                            vis_heatmap=False,
                            vis_grasp=False)
        # Ensure all device operations are completed before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Calculate and report the average inference time in milliseconds
    avg_time_ms = (time() - start) / T * 1e3
    print(f'Average inference time: {avg_time_ms:.2f} ms')
    print(f'Equivalent FPS: {1000/avg_time_ms:.2f}')
