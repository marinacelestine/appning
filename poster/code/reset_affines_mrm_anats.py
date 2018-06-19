import os
import glob
from sammba.registration import template_registrator
from sammba.registration.utils import _reset_affines
from sammba.externals.nipype.utils.filemanip import fname_presuffix
from sammba.externals.nipype.interfaces import afni

refit = afni.Refit().run
                  
anat_files = glob.glob(os.path.expanduser(
    '~/mrm_bil2_transformed/bil2_transfo_C57*.nii.gz'))
anat_files.remove(os.path.expanduser(
    '~/mrm_bil2_transformed/bil2_transfo_C57_Az1_invivo.nii.gz'))
output_dir = os.path.join(os.path.expanduser('~/mrm_bil2_transformed_preprocessed'))
template_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_template_invivo.nii.gz')

# Correct the header
template_atlas_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_atlas_invivo.nii.gz')
correct_template_atlas_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/correct_headers/Average_atlas_invivo_corrected.nii.gz')
_reset_affines(template_atlas_file, correct_template_atlas_file, xyzscale=1, overwrite=True,
               center_mass=(0, 0, 0))

template_brain_mask_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_brain_mask_invivo.nii.gz')
correct_template_brain_mask_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/correct_headers/Average_brain_mask_invivo_corrected.nii.gz')

_reset_affines(template_brain_mask_file, correct_template_brain_mask_file, xyzscale=1,
               overwrite=True)
out_refit = refit(in_file=template_brain_mask_file, duporigin_file=correct_template_atlas_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})

correct_template_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/correct_headers/Average_template_invivo_corrected.nii.gz')
_reset_affines(template_file, correct_template_file, xyzscale=1., overwrite=True)
out_refit = refit(in_file=correct_template_atlas_file, duporigin_file=correct_template_atlas_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
for anat_file in anat_files:
    correct_anat_file = fname_presuffix(anat_file, prefix='correct_headers/',
                                        suffix='_corrected')

    atlas_file = anat_file.replace('C57', 'Atlas')
    if 'y81' in anat_file:
        raw_atlas_file = atlas_file.replace('Invivo', 'invivo')

    correct_atlas_file = fname_presuffix(atlas_file, prefix='correct_headers/',
                                         suffix='_corrected')
    _reset_affines(atlas_file, correct_atlas_file, xyzscale=1, center_mass=(0, 0, 0),
                   overwrite=True)
    _reset_affines(anat_file, correct_anat_file, xyzscale=1, overwrite=True)
    out_refit = refit(in_file=correct_anat_file, duporigin_file=correct_atlas_file,
                      environ={'AFNI_DECONFLICT':'OVERWRITE'})
