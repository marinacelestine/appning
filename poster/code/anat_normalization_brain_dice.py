"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import numpy as np
from skimage import measure
from nilearn import image, masking
from nilearn._utils.niimg_conversions import check_niimg
from sammba.externals.nipype.interfaces import afni
from sammba.externals.nipype.caching import Memory
from sammba.externals.nipype.utils.filemanip import fname_presuffix


def compute_mask_contour(mask_file, write_dir=None, out_file=None):
    mask_img = check_niimg(mask_file)
    vertices, _ = measure.marching_cubes(mask_img.get_data(), 0)  #marching_cubes_lewiner
    vertices_minus = np.floor(vertices).astype(int)
    vertices_plus = np.ceil(vertices).astype(int)
    contour_data = np.zeros(mask_img.shape)
    contour_data[vertices_minus.T[0],
                 vertices_minus.T[1],
                 vertices_minus.T[2]] = 1
    contour_data[vertices_plus.T[0],
                 vertices_plus.T[1],
                 vertices_plus.T[2]] = 1
    contour_img = image.new_img_like(mask_img, contour_data)
    if write_dir is None:
        write_dir = os.getcwd()

    if out_file is None:
        out_file = fname_presuffix(mask_file, suffix='_countour',
                                   newpath=write_dir)
    contour_img.to_filename(out_file)
    return out_file

def dice(mask_file1, mask_file2):
    mask_data1 = check_niimg(mask_file1).get_data() > 0
    mask_data2 = check_niimg(mask_file2).get_data() > 0
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

if __name__ == '__main__':
    template_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma',
                                'templates/mouse/reoriented')
    template_brain_mask_file = os.path.join(template_dir,
                                            'brain100_binarized.nii')
    template_brain_contour_file = compute_mask_contour(
        template_brain_mask_file)
    spm_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes',
                           'Salma/inhouse_mouse_perf',
                           'mouse_091220/reoriented')
    sammba_dir = os.path.join('/home/bougacha',
                              'inhouse_mouse_perf_to_reoriented_head100',
                              'mouse_091220')

    # Create a precise brain mask, by combining RATS and SPM
    sammba_brain_mask_file = os.path.join(sammba_dir,
                                          'anat_n0_unifized_brain_mask.nii.gz')
    spm_tissues_files = [
        os.path.join(
            spm_dir, 'c{0}anat_n0_clear_hd.nii'.format(n))  # Try after N3
        for n in range(1, 4)]
    rough_mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .01', imgs=spm_tissues_files)
    spm_labels_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=rough_mask_img,
        imgs=spm_tissues_files)
    spm_gm_wm_img = image.math_img('np.logical_or(img==1, img==2)',
                                   img=spm_labels_img)
    brain_mask_img = masking.intersect_masks([sammba_brain_mask_file,
                                              spm_gm_wm_img],
                                             threshold=0)
    brain_mask_file = os.path.join(sammba_dir,
                                   'anat_n0_precise_brain_mask.nii.gz')
    brain_contour_file = compute_mask_contour(brain_mask_file)
    check_niimg(brain_contour_file).to_filename(
        os.path.join(spm_dir, 'anat_n0_precise_brain_mask_contour.nii'))

    nwarp_apply = afni.NwarpApply().run
    transforms = [
        os.path.join(sammba_dir,
                     'anat_n0_unifized_affine_general_warped_WARP.nii.gz'),
        os.path.join(sammba_dir, 'anat_n0_unifized_masked_aff.aff12.1D')]
    warp = "'" + ' '.join(transforms) + "'"
    sammba_registered_contour_file = fname_presuffix(brain_contour_file,
                                                     suffix='_warped',
                                                     newpath=sammba_dir)
    out_warp_apply = nwarp_apply(in_file=brain_contour_file,
                                 master=template_brain_mask_file,
                                 warp=warp,
                                 interp='nearestneighbor',
                                 out_file=sammba_registered_contour_file,
                                 environ={'AFNI_DECONFLICT':'OVERWRITE'})

    spm_registered_contour_file = os.path.join(
        spm_dir, 'wanat_n0_precise_brain_mask_contour.nii')

    spm_registered_contour_data = check_niimg(spm_registered_contour_file).get_data()
    template_shape = check_niimg(template_brain_mask_file).shape
    if spm_registered_contour_data.shape != template_shape:
        uncropped_spm_registered_brain_data = np.vstack(
            (np.zeros((1,) + template_shape[1:]), spm_registered_contour_data))
        uncropped_spm_registered_brain_data = np.nan_to_num(
            uncropped_spm_registered_brain_data)
        uncropped_spm_registered_contour_file = fname_presuffix(
            spm_registered_contour_file, suffix='_uncropped')
        image.new_img_like(
            spm_registered_contour_file,
            uncropped_spm_registered_brain_data).to_filename(
                uncropped_spm_registered_contour_file)
            
    else:
        uncropped_spm_registered_contour_file = spm_registered_contour_file

    for (contour_file, label) in zip([sammba_registered_contour_file,
                                      uncropped_spm_registered_contour_file],
                                      ['sammba', 'spm']):
        print('Brain contour DICE for ' + label,
              dice(template_brain_contour_file, contour_file))