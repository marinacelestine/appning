import os
import glob
from sammba.registration import template_registrator
from sammba.registration.utils import _reset_affines
from sammba.externals.nipype.utils.filemanip import fname_presuffix
from sammba.externals.nipype.interfaces import afni
import nibabel

            
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
template_brain_mask_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_brain_mask_invivo.nii.gz')
correct_template_brain_mask_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/correct_headers/Average_brain_mask_invivo_corrected.nii.gz')
correct_template_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/correct_headers/Average_template_invivo_corrected.nii.gz')
for (in_file, out_file) in zip([template_file, template_brain_mask_file, template_atlas_file],
                                [correct_template_file, correct_template_brain_mask_file,
                                 correct_template_atlas_file]):
    img = nibabel.load(in_file)
    header = img.header.copy()
    qform = header.get_qform()
    header.set_sform(qform, 1)
    nibabel.Nifti1Image(img.get_data(), qform, header).to_filename(out_file)
        

for anat_file in anat_files:
    correct_anat_file = fname_presuffix(anat_file, prefix='correct_headers/',
                                        suffix='_corrected')

    atlas_file = anat_file.replace('C57', 'Atlas')
    if 'y81' in anat_file:
        raw_atlas_file = atlas_file.replace('Invivo', 'invivo')

    correct_atlas_file = fname_presuffix(atlas_file, prefix='correct_headers/',
                                         suffix='_corrected')
    for (in_file, out_file) in zip([atlas_file, anat_file],
                                    [correct_atlas_file, correct_anat_file]):
    
        img = nibabel.load(in_file)
        header = img.header.copy()
        qform = header.get_qform()
        qform[0, 0] *= -1
        header.set_qform(qform, 2)
        header.set_sform(qform, 2)
        nibabel.Nifti1Image(img.get_data(), qform, header).to_filename(out_file)
