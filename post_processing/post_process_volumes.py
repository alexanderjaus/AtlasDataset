import numpy as np
import nibabel as nib
import cc3d
import copy 
import os

import multiprocessing

from skspatial.objects import Plane, Points

from  metadata import sex_metadata

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, help="Directory containing .nii.gz files to be postprocessed")
parser.add_argument("--target_dir", type=str, help="Directory containing .nii.gz files to be postprocessed")
parser.add_argument("--nof_jobs", type=int, default=4, help="Nof jobs for parallel processing the files")
parser.add_argument("--metadata_file", type=str, help="Path to the metadata file", required=False)

import logging

relevant_left_right_pairs = {
    "kidney": (15, 16),
    "breast": (22, 23),
    "adrenal gland": (27, 28),
    "thyroid": (29, 30),
    "gluteus maximus": (32, 33),
    "gluteus medius": (34, 35),
    "gluteus minimus": (36,37),
    "iliopsoas": (38, 39),
    "autochthon": (40, 41),
    "costa 1": (67, 68),
    "costa 2": (69, 70),
    "costa 3": (71, 72),
    "costa 4": (73, 74),
    "costa 5": (75, 76),
    "costa 6": (77, 78),
    "costa 7": (79, 80),
    "costa 8": (81, 82),
    "costa 9": (83, 84),
    "costa 10": (85, 86),
    "costa 11": (87, 88),
    "costa 12": (89, 90),
    "clavicle": (93, 94),
    "scapula": (95, 96),
    "humerus": (97, 98),
    "hip": (100, 101),
    "femur": (103, 104),
    "iliac artery": (112, 113),
    "iliac vena": (115, 116)
}

vertebrae_labels = {
    "vertebrae C1": 43,
    "vertebrae C2": 44,
    "vertebrae C3": 45,
    "vertebrae C4": 46,
    "vertebrae C5": 47,
    "vertebrae C6": 48,
    "vertebrae C7": 49,
    "vertebrae T1": 50,
    "vertebrae T2": 51,
    "vertebrae T3": 52,
    "vertebrae T4": 53,
    "vertebrae T5": 54,
    "vertebrae T6": 55,
    "vertebrae T7": 56,
    "vertebrae T8": 57,
    "vertebrae T9": 58,
    "vertebrae T10": 59,
    "vertebrae T11": 60,
    "vertebrae T12": 61,
    "vertebrae L1": 62,
    "vertebrae L2": 63,
    "vertebrae L3": 64,
    "vertebrae L4": 65,
    "vertebrae L5": 66
}

# Do not add anything to this label
rib_labels = {
    "costa 1 left": 67,
    "costa 1 right": 68,	 
    "costa 2 left": 69,	
    "costa 2 right": 70,	 
    "costa 3 left": 71,
    "costa 3 right": 72,	 
    "costa 4 left": 73,
    "costa 4 right": 74,	 
    "costa 5 left" : 75,
    "costa 5 right": 76,	
    "costa 6 left": 77,	
    "costa 6 right": 78,	 
    "costa 7 left": 79,	
    "costa 7 right": 80,	 
    "costa 8 left": 81,	
    "costa 8 right": 82,	 
    "costa 9 left": 83,	
    "costa 9 right": 84,	 
    "costa 10 left": 85,	 
    "costa 10 right": 86,	
    "costa 11 left": 87, 
    "costa 11 right": 88,	
    "costa 12 left": 89,	 
    "costa 12 right": 90	 
}

vessel_labels = {
    "iliac artery left": 112,
    "iliac artery right": 113,
    "aorta": 114,
    "iliac vena left": 115,
    "iliac vena right": 116,
    "inferior vena cava": 117,
    "portal vein and splenic vein": 118,
    "celiac trunk": 119,
    "pulmonary artery":127,
    #TODO: Add new vessel labels
    "ARTERY_COMMONCAROTID_RECHTS": 133,
    "ARTERY_COMMONCAROTID_LINKS": 134,
    "ARTERY_INTERNALCAROTID_RECHTS": 136,
    "ARTERY_INTERNALCAROTID_LINKS": 137,
    "IJV_RECHTS": 138,
    "IJV_LINKS": 139,
    "ARTERY_BRACHIOCEPHALIC": 140,
    "VEIN_BRACHIOCEPHALIC_RECHTS": 141,
    "VEIN_BRACHIOCEPHALIC_LINKS": 142,
    "ARTERY_SUBCLAVIAN_RECHTS": 143,
    "ARTERY_SUBCLAVIAN_LINKS": 144
}

keep_fragment_labels = {
    "Background": 0,
    "Left to annotate": 1,
    "muscles": 2,
    "fat": 3,
    "abdominal tissue": 4,
    "mediastinal tissue": 5,
    "small bowel": 8,
    "duodenum": 9,
    "colon": 10,
    "gonads":18,
    #"uterocervix": 20,
    "breast right":22,
    "breast left": 23,
    "skin": 42,
    "rib_cartilage": 91,
    **vessel_labels,
    "heart tissue": 107,
    "heart": 105,
    "bronchie":125,
    "nasal cavity": 132,
}

head_processing_labels = {
    "small bowel": 8,
    "colon": 10,
    "gonads": 18,
    "rib_cartilage": 91,
}

abdomen_processing_labels = {
    "bladder": 17
}

above_lunge_procesing_labels = {
    "bronchie":125
}

with open("Data/label_name.csv","r") as f:
    labels = f.readlines()
all_labels = dict([line.replace("\n","").split(",") for idx, line in enumerate(labels) if idx > 0])
all_labels = {key:int(val) for val,key in all_labels.items()}

assumption = "The assumption for the postprocessing is the LAS orientation"

def get_index_arr(img):
    return np.moveaxis(np.moveaxis(np.stack(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))),0,3),0,1)

def count_ribs(img):
    """Check nof voxels in cc

    Args:
        head (_type_): _description_
        img (_type_): _description_
    """

    #Assumption is that 
    # x: sagital from right to left
    # y: coronal from back to front
    # z: axial from bottom to top

    logger = get_this_logger()
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)

    all_ribs_msk = np.zeros(img.shape).astype(np.bool8)
    
    for rib_name, rib_idx in rib_labels.items():
        all_ribs_msk = np.logical_or(all_ribs_msk, img == rib_idx)
    
    all_ribs_cc = cc3d.connected_components(all_ribs_msk, connectivity=6)
    uv, uc = np.unique(all_ribs_cc,return_counts=True)
    
    #Assumption is that ribs are not connected to each other. Keep only the 24 costa labels 
    all_rib_labels = uv[np.argsort(uc, )[::-1][1:1+len(rib_labels)]]
    median_points = []
    min_points = []

    for cc in all_rib_labels:
        median_points.append((np.median(index_arr[all_ribs_cc==cc], axis=0),cc))
        min_points.append((np.min(index_arr[all_ribs_cc==cc], axis=0),cc))
    
    rl_ordered_by_median = [x[1] for x in sorted(median_points, key=lambda x: x[0][0])]
    rl_ordered_by_min = [x[1] for x in sorted(min_points, key=lambda x: x[0][0])]

    assert set(rl_ordered_by_median[:12]) == set(rl_ordered_by_min[:12]), "Checking righ left cc by median and minimum values led to an different set of ribs"
    
    ribs_right, ribs_left = rl_ordered_by_median[:12], rl_ordered_by_median[12:]

    #Counting the left ribs from bottom to top
    median_points_left = []
    max_points_left = []
    mean_points_left = []
    for cc in ribs_left:
        median_points_left.append((np.median(index_arr[all_ribs_cc==cc], axis=0),cc))
        max_points_left.append((np.max(index_arr[all_ribs_cc==cc], axis=0),cc))
        mean_points_left.append((np.mean(index_arr[all_ribs_cc==cc], axis=0), cc))

    median_points_left = sorted(median_points_left, key=lambda x: x[0][2])
    max_points_left = sorted(max_points_left, key=lambda x:x[0][2])
    mean_points_left = sorted(mean_points_left, key=lambda x:x[0][2])

    
    if not all([x[1] == y[1] for x,y in zip(median_points_left, max_points_left)]): 
        logger.warning("Order induced by \
        maximum voxel is different for order induced by median voxels")

    median_points_right = []
    max_points_right = []
    mean_points_right = []
    for cc in ribs_right:
        median_points_right.append((np.median(index_arr[all_ribs_cc==cc], axis=0),cc))
        max_points_right.append((np.min(index_arr[all_ribs_cc==cc], axis=0), cc))
        mean_points_right.append((np.mean(index_arr[all_ribs_cc==cc], axis=0), cc))

    median_points_right = sorted(median_points_right, key=lambda x:x[0][2])
    max_points_right = sorted(max_points_right, key=lambda x: x[0][2])
    mean_points_right = sorted(mean_points_right, key=lambda x:x[0][2])

    if not all([x[1] == y[1] for x,y in zip(median_points_right, max_points_right)]): 
        logger.warning("Order induced by maximum voxel is different for order induced by median voxels")
    if not all([x[1]==y[1] for x,y in zip(median_points_right, mean_points_right)]):
        logger.warning("Order induced by median values is different from order induced by mean values")

    # Assign ribs
    ordered_left_rib_labels = sorted([
        label for name, label in rib_labels.items() if "left" in name
    ])

    ordered_right_rib_labels = sorted([
        label for name, label in rib_labels.items() if "right" in name
    ])

    for i in range(len(mean_points_left)):
        costa_label_left, cc_identifier_left = ordered_left_rib_labels[i], mean_points_left[i][1]
        costa_label_right, cc_identifier_right = ordered_right_rib_labels[i], mean_points_right[i][1]
        
        msk_left = all_ribs_cc == cc_identifier_left
        msk_right = all_ribs_cc == cc_identifier_right
        
        costa_voxel_left = index_arr[msk_left]
        costa_voxel_right = index_arr[msk_right]

        for voxel in costa_voxel_left:
            img_mod[tuple(voxel)] = costa_label_left
        
        for voxel in costa_voxel_right:
            img_mod[tuple(voxel)] = costa_label_right
    
    return img_mod

def merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold):
    
    
    nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
    relevant_cc = []

    for idx, nof_voxels in nof_voxels_cc:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc.append((idx, nof_voxels))
    
    #Remove background cc from relevant cc. Assumption is that background is largest cc
    relevant_cc = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[1:]
    
    nof_voxels_above = [(x, np.sum(cc_above == x )) for x in np.unique(cc_above)]

    relevant_cc_above = []
    for idx, nof_voxels in nof_voxels_above:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc_above.append((idx, nof_voxels))
        # Do not supress small components here, as they will be handeled at the vertebra itself
    
    #Ignore the largest non background_cc component as it well be the vertebra itself
    relevant_cc_above = sorted(relevant_cc_above, key=lambda x: x[1], reverse=True)[2:]

    #There are components left from the vertebra which are neither background nor the vertebra itself
    if len(relevant_cc_above) > 0:
        #Pool the remaining components above with all relevant cc of current vertebra 
        mskcc_pool = np.zeros(cc_cur.shape).astype(np.bool8)
        for idx, _ in relevant_cc_above:
            mskcc_pool = np.logical_or(mskcc_pool, cc_above==idx)
        for idx, _ in relevant_cc:
            mskcc_pool = np.logical_or(mskcc_pool, cc_cur == idx)

        cc_pool = cc3d.connected_components(mskcc_pool)
        rel_components_pool = sorted([(x, np.sum(cc_pool == x )) for x in np.unique(cc_pool)],key=lambda x:x[1], reverse=True)[1:]

        return cc_pool==rel_components_pool[0][0]
    
    else:
        return None

def get_relevant_ccs(cc, keep_threshold, keep_main=True):
    if keep_main:
        cutoff_idx = 1
    else:
        cutoff_idx = 2
    return sorted([(x,np.sum(cc==x)) for x in np.unique(cc) if np.sum(cc==x) > keep_threshold],key=lambda x:x[1], reverse=True)[cutoff_idx:]


def spine_adjacent_pairs(img, voxel_supression_threshold=10, default_val=0, include_sacrum=True):
    """Check alternating connected component to identfy fractins assigned to the wrong vertebra"""
    labels = list(vertebrae_labels.values())
    if include_sacrum:
        labels.append(102)
    
    mod_img = copy.deepcopy(img)

    #Get triplets of adjacent vertebras
    triplets = []
    for l in range(len(labels)):
        # Regular triplet
        if l > 0 and l < len(labels)-1:
            triplets.append((labels[l-1], labels[l], labels[l+1]))
        # First triplet
        elif l<len(labels)-1:
            assert l == 0, "Just to be sure" #TODO: Remove before release
            triplets.append((labels[l], labels[l+1]))
        # Last triplet
        elif l>0:
            assert l==len(labels)-1, "Just to be sure" #TODO: Remove before release
            triplets.append((labels[l-1], labels[l]))
    
    for idx, triplet in enumerate(triplets):
        #Seperately handel first and last triplet
        if idx==0 or idx==len(triplets)-1:
            current, below = triplet
            above = None
        elif idx == len(triplets)-1:
            above, current = triplet
            below = None
        #Standard triplet
        else:
            above, current, below = triplet
            
            msk_cur = mod_img == current
            cc_cur = cc3d.connected_components(msk_cur)
            
            #Supress small connectred components
            nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]

            relevant_cc = []

            for idx, nof_voxels in nof_voxels_cc:
                if nof_voxels > voxel_supression_threshold:
                    relevant_cc.append((idx, nof_voxels))
                else:
                    #Set fragments smaller than voxel_supression_threshold to background
                    mod_img[cc_cur == idx] = default_val
            
            #Remove background cc from relevant cc. Assumption is that background is largest cc
            background_index = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[0]
            relevant_cc.remove(background_index)

            if above is not None:
                msk_above = mod_img == above
                cc_above = cc3d.connected_components(msk_above, connectivity=6)
                rel_cc_above = get_relevant_ccs(cc_above,keep_threshold=voxel_supression_threshold, keep_main=False)
            
            if below is not None:
                msk_below = mod_img == below
                cc_below = cc3d.connected_components(msk_below, connectivity=6)
                rel_cc_below = get_relevant_ccs(cc_below,keep_threshold=voxel_supression_threshold, keep_main=False)
            
            if above is not None and len(rel_cc_above) > 0:
                
                consolidated_vetebra_above = merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold=voxel_supression_threshold)
                if consolidated_vetebra_above is not None:
                    mod_img[consolidated_vetebra_above] = current
                     
            
            elif below is not None and len(rel_cc_below) > 0:
                consolidated_vetebra_below = merge_cc_of_adjacent(cc_cur, cc_below, voxel_supression_threshold=voxel_supression_threshold)
                if consolidated_vetebra_below is not None:
                    mod_img[consolidated_vetebra_below] = current
    return mod_img    
    
def sample_random_points(points, sample_size=5000):
    """Sample random points without duplicates"""
    choice = np.random.permutation(np.arange(points.shape[0]))[:sample_size]
    return points[choice]

def split_right_left(img, nondfault_labels_to_split=None, filename=None):
    """
    Regress a hpyerplane through the median of the vertebras, sacrum and the the sternum. Check if left_right predicted labels
    agree with this geometric view.
    Args:
        img (_type_): _description_
        labels (_type_): _description_
    """
    logger = get_this_logger()
    
    spine_labels = list(vertebrae_labels.values())
    sternum_labels = [92]
    sacrum_label = [102]

    img_mod = copy.deepcopy(img)

    median_points = []
    index_arr = get_index_arr(img)
    
    for label in spine_labels + sternum_labels + sacrum_label:
        msk = img == label
        points = index_arr[msk]
        #median_values = median_values * header["pixdim"][1:4]
        median = np.median(points, axis=0)
        if not any(np.isnan(median)):
            median_points.append(median)
    
    if len(median_points)!=len(vertebrae_labels)+len(sternum_labels)+len(sacrum_label):
        exception_logger = get_error_logger()
        exception_logger.info(f"Problematic file {filename}")

    #Regress hyperplane through the pointcoulds 
    assert len(median_points) >= 2, f"Need to predict at least 2 vertebtra predictions to regress hyperplane, but only got {len(median_points)}"
    median_points = Points(median_points)
    plane = Plane.best_fit(median_points)
    
    normal_vector = plane.vector
    point_plane = plane.point

    assert 0.9 < np.abs(normal_vector[0]) < 1.1, "Large value for normal vector of hyperplane"

    #Define left and right side of plane relative. Assumption is that 
    # x: sagital from right to left
    # y: coronal from back to front
    # z: axial from bottom to top

    if np.sign(normal_vector[0]) == -1:
        left = 1
        right = -1
    else:
        left = -1
        right = 1

    #Double check r/l with liver (label 13) prediction which has to be on the right body side
    if np.sign(np.dot(point_plane-np.median(index_arr[img == 13],axis=0), normal_vector)) != right:
        logger.warning("WARNING: LABEL LIVER IS NOT ON THE RIGHT SIDE OF THE BODY ",
        "This may be either due to a very bad liver predictoin or wrong assignment of hyperplan")
    
    #Merge confused right and left organs
    if nondfault_labels_to_split is not None:
        left_right_pairs = nondfault_labels_to_split
    else:
        left_right_pairs = relevant_left_right_pairs
    
    for name, pair in left_right_pairs.items():
        logger.info(f"Processing class {name}")

        label_left, label_right = pair

        if name not in ["iliac artery", "iliac vena"]:
            points_labeled_left = index_arr[img == label_left]
            points_labeled_right = index_arr[img == label_right]
            
            #Check which points on the right on the hyperplane are assigned as label_left and assign them as label_right
            left_points_geometric_check = np.sign(np.dot(point_plane - points_labeled_left, normal_vector))
            left_right_confusion = left_points_geometric_check == right
            if left_right_confusion.sum().item() > 0:
                wrong_voxels = points_labeled_left[left_right_confusion]
                for voxel in wrong_voxels:
                    img_mod[tuple(voxel)] = label_right

            #Check which points on the left of the hyperplane are assigned as label_right and assign them as label_left
            right_points_geometric_check = np.sign(np.dot(point_plane - points_labeled_right, normal_vector))
            right_left_confusion = right_points_geometric_check == left
            if right_left_confusion.sum().item()>0:
                wrong_voxels = points_labeled_right[right_left_confusion]
                for voxel in wrong_voxels:
                    img_mod[tuple(voxel)] = label_left
            
            #Merge labels close to hyperplane to largest cc, as they might have been cut close to the hyperplane
            cc_left = cc3d.connected_components(img_mod == label_left)
            cc_right = cc3d.connected_components(img_mod == label_right)

            consolidated_left = merge_cc_of_adjacent(cc_left, cc_right, voxel_supression_threshold=5)
            if consolidated_left is not None:
                img_mod[consolidated_left] = label_left
            
            consolidated_right = merge_cc_of_adjacent(cc_right, cc_left, voxel_supression_threshold=5)
            if consolidated_right is not None:
                img_mod[consolidated_right] = label_right
        
        else:
            #Only look at venas below the upper limit of the sacrum, as before is too close to hyperplane
            sacrum_cc = cc3d.connected_components(img_mod==sacrum_label[0])
            sacrum_cc_identifier, sacrum_cc_count = np.unique(sacrum_cc, return_counts=True)
            sacrum_main_cc_index = sacrum_cc_identifier[np.argsort(sacrum_cc_count)[::-1][1]]
            sacrum_main_cc_max = np.max(index_arr[sacrum_cc == sacrum_main_cc_index], axis=0)
            
            for vessle_pairs in ["iliac artery", "iliac vena"]:
                msk_left = img_mod == left_right_pairs[vessle_pairs][0]
                msk_left[:,:,:sacrum_main_cc_max[2]] = False
                cc_left = cc3d.connected_components(msk_left, connectivity=6)
                msk_right = img_mod == left_right_pairs[vessle_pairs][1]
                msk_right[:,:,:sacrum_main_cc_max[2]] = False
                cc_right = cc3d.connected_components(msk_right, connectivity=6)
                consolidated_left = merge_cc_of_adjacent(cc_left, cc_right, voxel_supression_threshold=5)
                if consolidated_left is not None:
                    img_mod[consolidated_left] = left_right_pairs[vessle_pairs][0]
                consolidated_right = merge_cc_of_adjacent(cc_right, cc_left, voxel_supression_threshold=5)
                if consolidated_right is not None:
                    img_mod[consolidated_right] = left_right_pairs[vessle_pairs][1]
        
    return img_mod
        
def supress_non_largest_components(img, default_val = 0):
    """As a last step, supress all non largest components"""
    logger = get_this_logger()
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)
    new_background = np.zeros(img.shape, dtype=np.bool8)
    for name, label in all_labels.items():
        if name not in keep_fragment_labels:
            logger.info(f"Supressing nonlargest cc for label {name}")
            label_cc = cc3d.connected_components(img == label, connectivity=6)
            uv, uc = np.unique(label_cc, return_counts=True)
            dominant_vals = uv[np.argsort(uc)[::-1][:2]]
            if len(dominant_vals)>=2: #Case: no predictions
                new_background = np.logical_or(new_background, np.logical_not(np.logical_or(label_cc==dominant_vals[0], label_cc==dominant_vals[1])))
    for voxel in index_arr[new_background]:
        img_mod[tuple(voxel)] = default_val
    return img_mod

def split_cc_based_on_axial_locatoin(img):
    """Create new labels by consisting misspredictions"""
    #TODO: Make more general: Give multiple cc prediction

    index_arr = get_index_arr(img)
    logger = get_this_logger()
    img_mod = copy.deepcopy(img)

    #Assumption is that 
    # x: sagital from right to left
    # y: coronal from back to front
    # z: axial from bottom to top

    #Case 1 cheeks
    logger.info("Assigning upper thyroid predictions at skull as cheeks")
    new_label_left = 128
    new_label_right = 129

    thyroid_mask_left = cc3d.connected_components(img == relevant_left_right_pairs["thyroid"][0])
    uv_left, uc_left = np.unique(thyroid_mask_left, return_counts=True)
    rel_vals_left = uv_left[np.argsort(uc_left)[::-1][1:3]]
    
    thyroid_mask_right = cc3d.connected_components(img == relevant_left_right_pairs["thyroid"][1])  
    uv_right, uc_right = np.unique(thyroid_mask_right ,return_counts=True)
    rel_vals_right = uv_right[np.argsort(uc_right)[::-1][1:3]]

    left_medians = []
    right_medians = []
    for val in rel_vals_left:
        left_medians.append((np.median(index_arr[thyroid_mask_left==val], axis=0), val))
    
    for val in rel_vals_right:
        right_medians.append((np.median(index_arr[thyroid_mask_right==val], axis=0), val))

    if len(left_medians)==2:
        lower_left, upper_left = sorted(left_medians, key=lambda x: x[0][2])
        upper_val_left = upper_left[1]
        lower_val_left = lower_left[1]

        cheek_left_msk = thyroid_mask_left == upper_val_left
        img_mod[cheek_left_msk] = new_label_left
    
    if len(right_medians)==2:
        lower_right, upper_right = sorted(right_medians, key=lambda x:x[0][2])
        upper_val_right = upper_right[1]
        lower_val_right = lower_right[1]

        cheek_right_msk = thyroid_mask_right == upper_val_right
        img_mod[cheek_right_msk] = new_label_right
        
    return img_mod

def bodypart_limitations(img):

    logger = get_this_logger()

    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)

    """Partition Body into certrain regions"""
    """Estimating skull as upper of median points of the two lowest Cervical vertebra"""
    low_head_vertebra_labels = {
                            "vertebrae C6": 48,
                            "vertebrae C7": 49
                        }
    median_points_low_head = []
    for vertebra, label in low_head_vertebra_labels.items():
        median_points_low_head.append(np.median(index_arr[label == img],axis=0))
    mean_median_position_c67 = np.mean(np.array(median_points_low_head),axis=0)

    # Easiest appraoch: Determine z-coodinate to separate head from the body
    boundary_head = mean_median_position_c67[2]
    logger.info(f"Axial values larger than {boundary_head} will be interpreted as part of the head")
    
    #TODO: Estimate boundary upper head
    high_head_vertebra_labels = {
         "vertebrae C1": 43
    }
    median_points_high_head = []
    for vertebra, label in high_head_vertebra_labels.items():
        median_points_high_head.append(np.median(index_arr[img_mod == label], axis=0))
    mean_median_position_c1 = np.mean(np.array(median_points_high_head), axis=0)
    boundary_high_head = mean_median_position_c1[2]
    logger.info(f"Axial values larger than {boundary_high_head} will be interpreted as part of the upper head")

    """Estimate lumbar as the body below the following vertebras"""
    lumber_vertebra_labels = {
            "vertebrae L3": 64,
            "vertebrae L4": 65,
            "vertebrae L5": 66
    }
    median_points_lumber = []

    for vertebra, label in lumber_vertebra_labels.items():
        median_points_lumber.append(np.median(index_arr[label == img], axis=0))
    mean_median_position_l345 = np.mean(np.array(median_points_lumber), axis=0)

    boundary_abdomen = mean_median_position_l345[2]
    logger.info(f"Axial values larger than {boundary_abdomen} will be interpreted as part of abdomen")

    #TODO: Make more general, pass list of acceptable labels per body region
    #TODO: Check remapping of values
    label_head_rules = {
        "rib_cartilage": lambda x: 1,
        "gonads": lambda x: 130,
        "colon": lambda x: 132 if x[2]>boundary_high_head else 1,
        "small bowel": lambda x: 132 if x[2]>boundary_high_head else 1,
        "bladder": lambda x: 1
    }

    label_abdomen_rules = {
        "bladder": lambda x: 12
    }

    upper_lunge_rules = {
        "bronchie": lambda x: 126
    }

    lung_box_upper = {
        'lung upper lobe left': 121, 
        'lung upper lobe right': 124
    }
    
    lung_box_lower = {
         'lung lower lobe left': 120, 
         'lung lower lobe right': 122,
    }
    
    max_points_upper_lunge = []
    min_points_lower_lunge = []

    for name, label in lung_box_upper.items():
        ccs = cc3d.connected_components(img_mod==label)
        uv, uc = np.unique(ccs,return_counts=True)
        main_cc_index = np.argsort(uc)[::-1][1]
        msk = ccs == uv[main_cc_index] 
        max_points_upper_lunge.append(np.max(index_arr[msk], axis=0))
    
    mean_max_points_upper_lunge = np.mean(max_points_upper_lunge,axis=0)
    upper_bound_lung = mean_max_points_upper_lunge[2]
    logger.info(f"Interpreting values larger than {upper_bound_lung} as above lunge")

    for name, label in lung_box_lower.items():
        ccs = cc3d.connected_components(img_mod==label)
        uv, uc = np.unique(ccs,return_counts=True)
        main_cc_index = np.argsort(uc)[::-1][1]
        msk = ccs == uv[main_cc_index] 
        min_points_lower_lunge.append(np.min(index_arr[msk], axis=0))
    
    mean_min_points_lower_lunge = np.mean(min_points_lower_lunge,axis=0)
    lower_bound_lung = mean_min_points_lower_lunge[2]
    logger.info(f"Interpreting values smaller than {lower_bound_lung} as below lung")

  
    # Assign values larger than the lung according to the rules
    for name, label in above_lunge_procesing_labels.items():
        msk = img_mod == label
        for voxel in index_arr[msk]:
            if voxel[2]>upper_bound_lung:
                img_mod[tuple(voxel)] = upper_lunge_rules[name](voxel)

    #Right now, only supress colon and rib cartige and remap gonads as eyes
    for name, label in head_processing_labels.items():
        msk = img_mod == label
        for voxel in index_arr[msk]:
            if voxel[2]> boundary_head:
                img_mod[tuple(voxel)] = label_head_rules[name](voxel)
    
    for name, label in abdomen_processing_labels.items():
        msk = img_mod == label
        for voxel in index_arr[msk]:
            if boundary_abdomen < voxel[2] < boundary_head:
                #Voxel is inside abdomen
                img_mod[tuple(voxel)] = label_abdomen_rules[name](voxel)
            elif voxel[2]>boundary_head:
                img_mod[tuple(voxel)] = label_head_rules[name](voxel)
    #Separate predictions for gonads into left and right 
    img_mod = split_right_left(img_mod, nondfault_labels_to_split={"eyeballs":(130,131)})    

    return img_mod

def sex_based_processing(img, img_name):
    logger = get_this_logger()
    img_mod = copy.deepcopy(img)
    identifier = img_name.split("_")[1]
    if identifier in sex_metadata:
        sex = sex_metadata[identifier]
        if sex == "M":
            #Ignore cervix preditions
            img_mod[img_mod==20] = 0
            #Ignore predictions for uterus
            img_mod[img_mod==21] = 0
        elif sex == "F":
            #Ignore predictions for prostate
            img_mod[img_mod==19] = 0
        else:
            logger.warning(f"Received an unexpected value for sex {sex}")
    else:
        logger.warning(f"Could not determine sex for file {img_name}. No processing applied")
    return img_mod
        

def process_file(args):
    file_path, target_path = args
    file_name = file_path.split("/")[-1]
    nib_img = nib.load(file_path)
    img, header = nib_img.get_fdata().astype(np.uint8), nib_img.header
    
    logger = get_this_logger()
    logger.info(f"Now processing file {file_name}")

    try:
        img = split_right_left(img, filename=file_name)
        logger.info(f"Completed right left splitting for file {file_name}")
        img = spine_adjacent_pairs(img, include_sacrum=True)
        logger.info(f"Completed spine processing for file {file_name}")
        img = count_ribs(img)
        logger.info(f"Completed rib processing for file {file_name}")
        img = split_cc_based_on_axial_locatoin(img)
        logger.info(f"Completed splic cc based on axial location for file {file_name}")
        img = bodypart_limitations(img)
        logger.info(f"Completed bodypart limitations for file {file_name}")
        img = sex_based_processing(img, file_name)
        logger.info(f"Completed sex based postprocessing for file {file_name}")
        img = supress_non_largest_components(img)
        logger.info(f"Completed supress non largest cc for file {file_name}")
        postprcessed_img = nib.Nifti1Image(img.astype(np.uint8), affine=nib_img.affine, header=header)
        nib.save(postprcessed_img, target_path)
        logger.info(f"Completed postprocessing for file {file_name}")
    except Exception as e:
        import traceback     

        logger.error(f"Got an error when processing file {file_name}")
        logger.error(e)
        logger.error(traceback.format_exc())

def get_this_logger(print_to_console=True):
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s | ')
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    handler = logging.FileHandler("logs/postprocessing.log")
    handler.setFormatter(formatter)
    if not len(logger.handlers):
        logger.addHandler(handler)
    if print_to_console and not len(logger.handlers)>1:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def get_error_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s | ')
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    handler = logging.FileHandler("logs/problematic_files.log")
    handler.setFormatter(formatter)
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger

def main(args):
    logger = get_this_logger()
    source_dir = args.source_dir
    target_dir = args.target_dir
    nof_jobs = args.nof_jobs

    source_files = list(filter(lambda x: x.endswith(".nii.gz"), os.listdir(source_dir)))
    nof_source_files = len(source_files)
    logger.info(f"Found a total of {nof_source_files} source images")
    #remove already existing files from the source file lists
    source_files = [x for x in source_files if not os.path.isfile(os.path.join(target_dir,x))]
    nof_consolidated_source_files = len(source_files)
    logger.info(f"Removed a total of {nof_source_files-nof_consolidated_source_files}. \
        {nof_consolidated_source_files} files will be post processed")

    logger.info(f"Start postprocessing with on {nof_jobs} jobs.")
    data_input = [(os.path.join(source_dir,file), os.path.join(target_dir,file)) for file in source_files]
    if nof_jobs>1:
        with multiprocessing.Pool(nof_jobs) as pool:
            pool.map(process_file, data_input)
    else:
        for inp in data_input:
            process_file(inp)
    logger.info("Finished Postprocessing")

if __name__ == "__main__":
    main(parser.parse_args())
