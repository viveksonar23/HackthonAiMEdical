import open3d as o3d
import numpy as np
import scipy.spatial as ss

total_volume_m3 = 0.0 
# Input path
input_file_path = "D://New folder//ImageToStl.com_result_image (12)_256.ply"
human_body_density_kg_m3 = 985  # kg/m³
# Load the original point cloud
pcd_original = o3d.io.read_point_cloud(input_file_path)

# Visualize the original point cloud
print("Visualizing the original point cloud...")
o3d.visualization.draw_geometries([pcd_original])

# Perform DBSCAN clustering to isolate significant clusters
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd_original.cluster_dbscan(eps=0.08, min_points=10, print_progress=True))

max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

# Ensure there is at least one cluster
if max_label < 0:
    raise ValueError("No clusters found in the point cloud.")

# Extract points from the chosen cluster
cluster_index = 0  # Choose the first cluster (or adjust based on the output)
cluster_indices = (labels == cluster_index)
cluster_pcd = o3d.geometry.PointCloud()
cluster_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_original.points)[cluster_indices])

# Verify cluster has points
if len(cluster_pcd.points) == 0:
    raise ValueError(f"Cluster {cluster_index} contains no points. Try adjusting the cluster index.")

# Visualize the clustered point cloud
print("Visualizing the clustered point cloud...")
o3d.visualization.draw_geometries([cluster_pcd])

# Process point cloud into a 3 x 100 grid (300 segments) along x and y axes
points = np.asarray(cluster_pcd.points)

# Define x and y limits for segmentation
X_min, X_max = np.min(points[:, 0]), np.max(points[:, 0])
Y_min, Y_max = np.min(points[:, 1]), np.max(points[:, 1])

# Generate 3 cuts along x-axis and 100 cuts along y-axis
X_limits = np.linspace(X_min, X_max, 4)    # 3 cuts -> 4 boundaries
Y_limits = np.linspace(Y_min, Y_max, 101)  # 100 cuts -> 101 boundaries

# Initialize list for 3 x 100 grid segments
point_segments = [[[] for _ in range(100)] for _ in range(3)]

# Sort points into the 3 x 100 grid segments
for point in points:
    x_cor, y_cor = point[0], point[1]
    x_index = np.searchsorted(X_limits, x_cor, side='right') - 1
    y_index = np.searchsorted(Y_limits, y_cor, side='right') - 1
    
    if x_index < 3 and y_index < 100:  # Ensure indices are within range
        point_segments[x_index][y_index].append(point)

# Convert segments to point clouds, compute hulls, add smaller gaps, and calculate volumes
pcd_segment_hulls = []
pcd_segment_volumes = []
pcd_segments_visualization = []

# Sum volumes of all segments
total_volume = np.sum(pcd_segment_volumes)
# print(f'Total Volume = {total_volume}')

weight_kg = total_volume * human_body_density_kg_m3
print(f"Total Body Volume: {total_volume:.6f} m³")
print(f"Estimated Weight: {weight_kg:.2f} kg")

# Visualization of segments and hulls
print("Visualizing segmented point clouds individually...")
o3d.visualization.draw_geometries(pcd_segments_visualization)

# print("Visualizing convex hulls of each segment individually...")
o3d.visualization.draw_geometries(pcd_segment_hulls)

# print("Visualizing all segments and their convex hulls together...")
o3d.visualization.draw_geometries(pcd_segment_hulls + pcd_segments_visualization)

# Estimate normals for the combined point cloud and visualize with convex hull
combined_pcd = o3d.geometry.PointCloud()
combined_points = np.concatenate([np.asarray(pcd.points) for pcd in pcd_segments_visualization], axis=0)
combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
combined_pcd.estimate_normals()

# Create convex hull for combined point cloud
hull, _ = combined_pcd.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))  # Red color for hull visualization

# Visualize the final combined point cloud with its convex hull
# print("Visualizing the combined point cloud with convex hull...")
o3d.visualization.draw_geometries([combined_pcd, hull_ls])
