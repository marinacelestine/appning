"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import numpy as np
from nilearn import image
from nilearn._utils.niimg_conversions import check_niimg
from sammba.externals.nipype.interfaces import afni
from sammba.externals.nipype.utils.filemanip import fname_presuffix


def dice(mask_file1, mask_file2):
    mask_data1 = check_niimg(mask_file1).get_data() > 0
    mask_data2 = check_niimg(mask_file2).get_data() > 0
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

if __name__ == '__main__':
    template_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma',
                                'templates/mouse/reoriented')
    template_file = os.path.join(template_dir, 'head100.nii')
    template_brain_mask_file = os.path.join(template_dir,
                                            'brain100_binarized.nii')
    spm_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes',
                           'Salma/inhouse_mouse_perf',
                           'mouse_091220/reoriented')
    sammba_dir = os.path.join('/home/bougacha',
                              'inhouse_mouse_perf_to_reoriented_head100',
                              'mouse_091220')

    template_tissues_imgs = [
        os.path.join(template_dir, 'c{0}head100.nii'.format(n))
        for n in range(1, 4)]
    template_labels_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=template_brain_mask_file,
        imgs=template_tissues_imgs)

    spm_labels_files = [
        os.path.join(output_dir, prefix + 'anat_n0_clear_hd_labeled.nii.gz')
        for (prefix, output_dir) in zip(['wc', 'c'], [spm_dir, sammba_dir])]
    for (prefix, spm_labels_file) in zip(['wc', 'c'], spm_labels_files):
        spm_tissues_imgs = [
            os.path.join(
                spm_dir, prefix + '{0}anat_n0_clear_hd.nii'.format(n))  # Try after N3
            for n in range(1, 4)]
        mask_img = image.math_img(
            'np.max(imgs, axis=-1) > .01', imgs=spm_tissues_imgs)
        spm_labels_img = image.math_img(
            'img * (np.argmax(imgs, axis=-1) + 1)',
            img=mask_img,
            imgs=spm_tissues_imgs)
        check_niimg(spm_labels_img, dtype=float).to_filename(spm_labels_file)


    nwarp_apply = afni.NwarpApply().run
    transforms = [
        os.path.join(sammba_dir,
                     'anat_n0_unifized_affine_general_warped_WARP.nii.gz'),
        os.path.join(sammba_dir, 'anat_n0_unifized_masked_aff.aff12.1D')]
    warp = "'" + ' '.join(transforms) + "'"
    sammba_tissues_files = []
    for tissue_file in spm_tissues_imgs:
        sammba_tissue_file = fname_presuffix(tissue_file,
                                             suffix='_warped',
                                             newpath=sammba_dir)
        out_warp_apply = nwarp_apply(in_file=tissue_file,
                                     master=template_file,
                                     warp=warp,
                                     out_file=sammba_tissue_file)
        sammba_tissues_files.append(sammba_tissue_file)

    sammba_brain_mask_file = '/home/bougacha/inhouse_mouse/brain100_binarized.nii.gz'
    mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .01', imgs=sammba_tissues_files)
    sammba_labels_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=mask_img,
        imgs=sammba_tissues_files)
    sammba_labels_file = sammba_tissue_file.replace('c3', 'labeled_c')
    sammba_labels_img.to_filename(sammba_labels_file)

    template_brain_mask_img = image.math_img('img>0', img=template_labels_img)
    for labels_file in [spm_labels_files[0], sammba_labels_file]:
        for label in [1, 2, 3]:
            tissue_template_img = image.math_img('img=={0}'.format(label),
                                                 img=template_labels_img)
            tissue_registered_img = image.math_img('img=={0}'.format(label),
                                                   img=labels_file)
            print(dice(tissue_template_img, tissue_registered_img))

        # Create brain mask
        brain_mask_img = image.math_img('img>0', img=labels_file)
        print('brain: ', dice(template_brain_mask_file, brain_mask_img))
        brain_mask_img.to_filename(labels_file.replace('labeled', 'brain_mask'))
