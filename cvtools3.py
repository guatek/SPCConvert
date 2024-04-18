# -*- coding: utf-8 -*-
"""
cvtools3 - image processing tools for plankton images

a simplified version for color conversion and database indexing only

compatible with python 3.9.1
"""
import os
from math import pi
import cv2
from skimage import morphology, measure, restoration
from skimage.filters import scharr, gaussian
import numpy as np
from scipy import ndimage

def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    output = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    output = output/np.sum(output)

    return output

# import raw image
def import_image(abs_path,filename,raw=True,bayer_pattern=cv2.COLOR_BAYER_RG2RGB):

    # Load and convert image as needed
    img_c = cv2.imread(os.path.join(abs_path,filename),cv2.IMREAD_UNCHANGED)
    if raw:
        img_c = cv2.cvtColor(img_c,bayer_pattern)

    return img_c

# convert image to 8 bit with or without autoscaling
def convert_to_8bit(img,auto_scale=True):

    # Convert to 8 bit and autoscale
    if auto_scale:
        minimum_img = np.min(img)
        range_img = np.max(img)-minimum_img
        result = np.float32(img)-minimum_img
        result[result<0.0] = 0.0
        if range_img!= 0:
            result = result/range_img

        img_8bit = np.uint8(255*result)
    else:
        img_8bit = np.unit8(img)

    return img_8bit


def intensity_features(img, obj_mask):
    res = {}

    # assume that obj_mask contains one connected component
    prop = measure.regionprops(obj_mask.astype(np.uint8), img)[0]
    res["mean_intensity"] = prop.mean_intensity

    intensities = prop.intensity_image[prop.image]
    res["median_intensity"] = np.median(intensities)
    res["std_intensity"] = np.std(intensities)
    res["perc_25_intensity"] = np.percentile(intensities, 25)
    res["perc_75_intensity"] = np.percentile(intensities, 75)

    centroid = np.array(prop.centroid)
    weighted_centroid = np.array(prop.weighted_centroid)
    displacement = weighted_centroid - centroid
    displacement_image = np.linalg.norm(displacement / img.shape)
    displacement_minors = np.linalg.norm(displacement) / prop.minor_axis_length
    res['mass_displace_in_images'] = displacement_image
    res['mass_displace_in_minors'] = displacement_minors

    res["moment_hu_1"] = prop.weighted_moments_hu[0]
    res["moment_hu_2"] = prop.weighted_moments_hu[1]
    res["moment_hu_3"] = prop.weighted_moments_hu[2]
    res["moment_hu_4"] = prop.weighted_moments_hu[3]
    res["moment_hu_5"] = prop.weighted_moments_hu[4]
    res["moment_hu_6"] = prop.weighted_moments_hu[5]
    res["moment_hu_7"] = prop.weighted_moments_hu[6]

    return res

# extract simple features and create a binary representation of the image
def quick_features(img,save_to_disk=False,abs_path='',file_prefix='',cfg = []):
    """
    :param img: 8-bit array
    """
    # Pull out some settings from cfg if available
    if cfg:
        min_obj_area = cfg.get('MinObjectArea',100)
        objs_per_roi = cfg.get('ObjectsPerROI',1)
        deconv = cfg.get("Deconvolve").lower() == 'true'
        edge_thresh = cfg.get('EdgeThreshold',2.5)
        use_jpeg = cfg.get("UseJpeg").lower() == 'true'
        raw_color = cfg.get("SaveRawColor").lower() == 'true'
        # Note BGR rather than RGB for opencv compat
        gains = [
            1.0 / cfg.get('BlueGain',1.0),
            1.0 / cfg.get('GreenGain',1.0),
            1.0 / cfg.get('RedGain',1.0),
        ]
        boost = cfg.get('Boost',1.0)
    else:
        min_obj_area = 100
        objs_per_roi = 1
        deconv = False
        use_jpeg = False
        raw_color = True
        edge_thresh = 2.5

    # Define an empty dictionary to hold all features
    features = {}

    features['rawcolor'] = np.copy(img)
    # compute features from gray image
    gray = np.uint8(np.mean(img,2))

    # edge-based segmentation
    edges_mag = scharr(gray)
    edges_med = np.median(edges_mag)
    edges_thresh = edge_thresh*edges_med
    edges = edges_mag >= edges_thresh
    edges = morphology.closing(edges,morphology.square(3))
    filled_edges = ndimage.binary_fill_holes(edges)
    edges = morphology.erosion(filled_edges,morphology.square(3))


    bw_img = edges

    # Compute morphological descriptors
    label_img = morphology.label(bw_img,background=0)
    props = measure.regionprops(label_img,gray)

    # clear bw_img
    bw_img = 0*bw_img

    props = sorted(props, reverse=True, key=lambda k: k.area)

    if len(props) > 0:

        # Init mask with the largest area object in the roi
        bw_img = (label_img)== props[0].label
        bw_img_all = bw_img.copy()

        # use only the features from the object with the largest area
        if len(props) > objs_per_roi:
            n_objs = objs_per_roi
        else:
            n_objs = len(props)

        for f in range(0,n_objs):

            if props[f].area > min_obj_area:
                bw_img_all = bw_img_all + ((label_img)== props[f].label)

        # Take the largest object area as the roi area
        # no average
        max_area = props[0].area
        max_maj = props[0].major_axis_length
        max_min = props[0].minor_axis_length
        max_or = props[0].orientation
        max_eccentricity = props[0].eccentricity
        max_solidity = props[0].solidity

        # Calculate intensity features only for largest
        features_intensity = intensity_features(gray, bw_img)
        features['intensity_gray'] = features_intensity

        features_intensity = intensity_features(img[::, ::, 0], bw_img)
        features['intensity_red'] = features_intensity

        features_intensity = intensity_features(img[::, ::, 1], bw_img)
        features['intensity_green'] = features_intensity

        features_intensity = intensity_features(img[::, ::, 2], bw_img)
        features['intensity_blue'] = features_intensity

        # Check for clipped image
        if np.max(bw_img_all) == 0:
            bw = bw_img_all
        else:
            bw = bw_img_all/np.max(bw_img_all)

        clip_frac = float(np.sum(bw[:,1]) +
                np.sum(bw[:,-2]) +
                np.sum(bw[1,:]) +
                np.sum(bw[-2,:]))/(2*bw.shape[0]+2*bw.shape[1])
        features['clipped_fraction'] = clip_frac

        # Save simple features of the object
        features['area'] = max_area
        features['minor_axis_length'] = max_min
        features['major_axis_length'] = max_maj
        if max_maj == 0:
            features['aspect_ratio'] = 1
        else:
            features['aspect_ratio'] = max_min/max_maj
        features['orientation'] = max_or
        features['eccentricity'] = max_eccentricity
        features['solidity'] = max_solidity
        features['estimated_volume'] = 4.0 / 3 * pi * max_maj * max_min * max_min


        # print "Foreground Objects: " + str(avg_count)

    else:

        features['clipped_fraction'] = 0.0

        # Save simple features of the object
        features['area'] = 0.0
        features['minor_axis_length'] = 0.0
        features['major_axis_length'] = 0.0
        features['aspect_ratio'] = 1
        features['orientation'] = 0.0
        features['eccentricity'] = 0
        features['solidity'] = 0
        features['estimated_volume'] = 0

    # Masked background with Gaussian smoothing, image sharpening, and
    # reduction of chromatic aberration

    # mask the raw image with smoothed foreground mask
    blurd_bw_img = gaussian(bw_img_all,3)
    img[:,:,0] = img[:,:,0]*blurd_bw_img 
    img[:,:,1] = img[:,:,1]*blurd_bw_img 
    img[:,:,2] = img[:,:,2]*blurd_bw_img 



    # Make a guess of the PSF for sharpening
    psf = make_gaussian(5, 3, center=None)

    # sharpen each color channel and then reconbine


    if np.max(img) == 0:
        img = np.float32(img)
    else:
        img = np.float32(img)/np.max(img)

    # Tweak colors
    for i in range(0,3):
        img[:,:,i] = img[:,:,i] * gains[i] * boost 
        img[:,:,i] = np.clip(img[:,:,i],0,1)

    if deconv:

        img[img == 0] = 0.0001
        img[:,:,0] = restoration.richardson_lucy(img[:,:,0], psf, 7)
        img[:,:,1] = restoration.richardson_lucy(img[:,:,1], psf, 7)
        img[:,:,2] = restoration.richardson_lucy(img[:,:,2], psf, 7)

    # Rescale image to uint8 0-255
    img_min = np.min(img)
    img_range = np.max(img)-img_min
    if img_range == 0:
        img = np.zeros(img.shape(),dtype=np.uint8)
    else:
        img = np.uint8(255*(img-img_min)/img_range)

    features['image'] = img
    features['binary'] = 255*bw_img_all

    # Save the binary image and also color image if requested
    if save_to_disk:

        # convert and save images

        # Raw color (no background removal)
        if use_jpeg:
            if raw_color:
                cv2.imwrite(os.path.join(abs_path,file_prefix+"_rawcolor.jpeg"),features['rawcolor'])
            # Save the processed image and binary mask
            cv2.imwrite(os.path.join(abs_path,file_prefix+".jpeg"),features['image'])
        else:
            if raw_color:
                cv2.imwrite(os.path.join(abs_path,file_prefix+"_rawcolor.png"),features['rawcolor'])
            # Save the processed image and binary mask
            cv2.imwrite(os.path.join(abs_path,file_prefix+".png"),features['image'])

        # Binary should also be saved png
        cv2.imwrite(os.path.join(abs_path,file_prefix+"_binary.png"),features['binary'])


    return features
