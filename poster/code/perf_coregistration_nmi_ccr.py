"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import numpy as np
from nilearn import image, masking
from sammba.externals.nipype.interfaces import afni
from sammba.externals.nipype.utils.filemanip import fname_presuffix
from sammba.externals.nipype.interfaces.afni import utils
from nose.tools import assert_less
import nibabel


if __name__ == '__main__':
    mouse_dir = 'mouse_191851'
    my_dir = os.path.join(
        '/home/bougacha/inhouse_mouse_perf_preprocessed_mine',
        mouse_dir)
    my_perf1 = os.path.join(
        my_dir,
        'perfFAIREPI_n2_M0_unbiased_perslice_oblique.nii.gz')
    perf_to_anat_transform = os.path.join(
        my_dir,
        'anat_n0_unifized_warped_oblique_anat_to_func.aff12.1D')
    my_anat = os.path.join(
        my_dir,
        'anat_n0_unifized.nii.gz')
    allineate = afni.Allineate().run
    out_allineate = allineate(
        in_file=my_perf1,
        master=my_anat,
        in_matrix=perf_to_anat_transform,
        out_file=fname_presuffix(my_perf1,
                                 suffix='_in_anat'),
        environ={'AFNI_DECONFLICT':'OVERWRITE'})
    my_perf = out_allineate.outputs.out_file
    my_perf_data = nibabel.load(my_perf).get_data()
    my_perf_mask_img = image.math_img('img > {}'.format(my_perf_data.min()),
                                      img=my_perf)
    my_perf_mask_data = my_perf_mask_img.get_data().astype(bool)
    my_perf_mask = fname_presuffix(my_perf, suffix='_mask')
    my_perf_mask_img.to_filename(my_perf_mask)

    spm_dir = os.path.join(
        '/home/Pmamobipet/0_Dossiers-Personnes',
        'Salma/inhouse_mouse_perf_preprocessed_mine',
        mouse_dir)
    spm_perf = os.path.join(
        spm_dir,
        'rperfFAIREPI_n2_M0_unbiased_clean_hd.nii')
    spm_anat = os.path.join(
        spm_dir,
        'anat_n0_unifized_clean_hd.nii')

    spm_perf_data = nibabel.load(spm_perf).get_data()
    spm_perf_mask_img = image.math_img(
        'img > {}'.format(np.nanmin(spm_perf_data)),
        img=my_perf)
    spm_perf_mask_data = spm_perf_mask_img.get_data().astype(bool)
    spm_perf_mask = fname_presuffix(spm_perf, suffix='_mask')
    spm_perf_mask_img.to_filename(spm_perf_mask)

    l = utils.LocalBistat().run
    l_out = l(in_file1=spm_anat, in_file2=spm_perf, mask_file=spm_perf_mask,
              neighborhood=('RECT', (-2, -2, -2)),
              stat=['crA', 'spearman', 'normuti'],
              environ={'AFNI_DECONFLICT':'OVERWRITE'})

    spm_cr_data = image.index_img(l_out.outputs.out_file, 0).get_data()
    spm_spearman_data = image.index_img(l_out.outputs.out_file, 1).get_data()
    spm_nmi_data = image.index_img(l_out.outputs.out_file, 2).get_data()

    l_out = l(in_file1=my_anat, in_file2=my_perf, mask_file=my_perf_mask,
              neighborhood=('RECT', (-2, -2, -2)),
              stat=['crA', 'spearman', 'normuti'],
              environ={'AFNI_DECONFLICT':'OVERWRITE'})
    my_cr_data = image.index_img(l_out.outputs.out_file, 0).get_data()
    my_spearman_data = image.index_img(l_out.outputs.out_file, 1).get_data()
    my_nmi_data = image.index_img(l_out.outputs.out_file, 2).get_data()

    assert_less(spm_spearman_data[spm_perf_mask_data].mean(),
                my_spearman_data[my_perf_mask_data].mean())
    assert_less(spm_cr_data[spm_perf_mask_data].mean(),
                my_cr_data[my_perf_mask_data].mean())
    assert_less(spm_nmi_data[spm_perf_mask_data].mean(),
                my_nmi_data[my_perf_mask_data].mean())
