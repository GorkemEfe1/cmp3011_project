import numpy as np

CORE_LANDMARK_INDICES = [
    1, 4, 168,          # Nose tip, nose bottom, nose bridge
    33, 133, 159, 145,  # Right eye corners, top, bottom
    362, 263, 386, 374, # Left eye corners, top, bottom
    52, 55, 70, 107,    # Left eyebrow points
    282, 285, 336, 300, # Right eyebrow points
    61, 291, 13, 14     # Mouth outer corners, upper lip, lower lip
]

def extract_geometric_features(flat_coords):
    """
    Converts 1434 coordinates into a hybrid feature set of 77 metrics:
    8 geometric ratios + 69 normalized spatial coordinates.
    """
    coords = np.array(flat_coords).reshape(478, 3)
    
    skull_anchor = coords[168] 
    face_scale = np.linalg.norm(coords[168] - coords[1]) + 1e-6
    
    mouth_height = np.linalg.norm(coords[13] - coords[14])
    mouth_width = np.linalg.norm(coords[78] - coords[308])
    mar = mouth_height / (mouth_width + 1e-6)
    
    left_ear = np.linalg.norm(coords[386] - coords[374]) / (np.linalg.norm(coords[362] - coords[263]) + 1e-6)
    right_ear = np.linalg.norm(coords[159] - coords[145]) / (np.linalg.norm(coords[33] - coords[133]) + 1e-6)
    avg_ear = (left_ear + right_ear) / 2.0
    
    left_brow_dist = np.linalg.norm(coords[52] - coords[386]) / face_scale
    right_brow_dist = np.linalg.norm(coords[282] - coords[159]) / face_scale
    inner_brow_dist = np.linalg.norm(coords[55] - coords[285]) / face_scale
    
    mouth_center = (coords[13] + coords[14]) / 2.0
    avg_corner_drop = ((coords[78][1] - mouth_center[1]) + (coords[308][1] - mouth_center[1])) / (2.0 * face_scale)
    nose_scrunch = np.linalg.norm(coords[13] - coords[168]) / face_scale
    
    left_inner_brow_lift = np.linalg.norm(coords[55] - coords[168]) / face_scale
    right_inner_brow_lift = np.linalg.norm(coords[285] - coords[168]) / face_scale
    avg_inner_brow_lift = (left_inner_brow_lift + right_inner_brow_lift) / 2.0
    
    lip_to_chin = np.linalg.norm(coords[14] - coords[152]) / face_scale

    ratios = [
        mar, left_ear, right_ear, avg_ear, left_brow_dist, right_brow_dist, 
        inner_brow_dist, avg_corner_drop, nose_scrunch, 
        avg_inner_brow_lift, lip_to_chin  
    ]
    
    normalized_spatial_coords = []
    for idx in CORE_LANDMARK_INDICES:
        rel_coord = (coords[idx] - skull_anchor) / face_scale
        normalized_spatial_coords.extend(rel_coord.tolist()) 
        
    return ratios + normalized_spatial_coords