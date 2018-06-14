import os
import glob
from sammba.registration import template_registrator


anat_files = glob.glob(os.path.expanduser(
    '~/mrm_bil2_transformed/bil2_transfo_C57*.nii.gz'))
anat_files.remove(os.path.expanduser(
    '~/mrm_bil2_transformed/bil2_transfo_C57_Az1_invivo.nii.gz'))
output_dir = os.path.join(os.path.expanduser('~/mrm_bil2_transformed_preprocessed'))
template_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_template_invivo.nii.gz')
template_brain_mask_file = os.path.expanduser(
    '~/nilearn_data/mrm_2010/Average_brain_mask_invivo.nii.gz')

for anat_file in anat_files:
    registrator = template_registrator.TemplateRegistrator(
        brain_volume=400,
        dilated_template_mask=None,
        output_dir=output_dir,
        template=template_file,
        caching=True,
        template_brain_mask=template_brain_mask_file,
        registration_kind='affine')

    registrator.fit_anat(anat_file)