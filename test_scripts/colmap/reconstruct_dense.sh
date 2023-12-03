# Exit immediately if a command exits with a non-zero status.
set -e

VID=$1  # Video ID

IMAGE_PATH="data/demo/images/$VID"
CAMERAS_PATH="colmap_models/sparse/$VID/sparse/0"
# CAMERAS_PATH="colmap_models/dense/$VID"
DENSE3D_PATH="colmap_models/dense3D/$VID/"

mkdir -p $DENSE3D_PATH

COLMAP_COMMAND="singularity exec --nv --bind /home --bind /data/demo/ /data/demo/shubham/colmap.sif colmap"
# COLMAP_COMMAND="colmap"

echo "colmap image_undistorter"
$COLMAP_COMMAND image_undistorter \
    --image_path $IMAGE_PATH \
    --input_path $CAMERAS_PATH \
    --output_path $DENSE3D_PATH \
    --output_type COLMAP \
    --copy_policy soft-link \
    --max_image_size 2000

echo "colmap patch_match_stereo"
$COLMAP_COMMAND patch_match_stereo \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0 
    # Extra parameter for 14_05
    # \
    # --PatchMatchStereo.depth_min 0.1 \
    # --PatchMatchStereo.depth_max 12

# echo "colmap stereo_fusion"
# $COLMAP_COMMAND stereo_fusion \
#     --workspace_path $DENSE3D_PATH \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path $DENSE3D_PATH/fused.ply

minpix=15
echo "colmap stereo_fusion (min pix$minpix)"
$COLMAP_COMMAND stereo_fusion \
    --workspace_path $DENSE3D_PATH \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.min_num_pixels $minpix \
    --output_path $DENSE3D_PATH/fused-minpix$minpix.ply

# echo "colmap poisson_mesher"
# $COLMAP_COMMAND poisson_mesher \
#     --input_path $DENSE3D_PATH/fused.ply \
#     --output_path $DENSE3D_PATH/fused-meshed-poisson-d10-t5.ply \
#     --PoissonMeshing.depth 10 \
#     --PoissonMeshing.num_threads 5 \
#     --PoissonMeshing.trim 5

echo "colmap poisson_mesher"
$COLMAP_COMMAND poisson_mesher \
    --input_path $DENSE3D_PATH/fused-minpix$minpix.ply \
    --output_path $DENSE3D_PATH/fused-minpix$minpix-meshed-poisson-d10-t5.ply \
    --PoissonMeshing.depth 10 \
    --PoissonMeshing.num_threads 5 \
    --PoissonMeshing.trim 5

echo "copying fused-minpix$minpix.ply to fused.ply"
cp $DENSE3D_PATH/fused-minpix$minpix.ply $DENSE3D_PATH/fused.ply
cp $DENSE3D_PATH/fused-minpix$minpix.ply.vis $DENSE3D_PATH/fused.ply.vis
echo "colmap delaunay_mesher"
$COLMAP_COMMAND delaunay_mesher \
    --input_path $DENSE3D_PATH \
    --output_path $DENSE3D_PATH/fused-minpix$minpix-meshed-delaunay-qreg5.ply \
    --DelaunayMeshing.quality_regularization 5
