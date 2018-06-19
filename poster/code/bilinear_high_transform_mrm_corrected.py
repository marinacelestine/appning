"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import glob
from sammba.externals.nipype.interfaces import afni, fsl
from sammba.externals.nipype.utils.filemanip import fname_presuffix


if __name__ == '__main__':
    sammba_raw_dir = os.path.expanduser('~/nilearn_data/mrm_2010/correct_headers')
    sammba_transformed_dir = os.path.expanduser('~/mrm_bil2_transformed/correct_headers')
    sammba_params_dir = os.path.expanduser('~/mrm_bil2_transformed')

    anat_files = glob.glob(os.path.expanduser(
        '~/nilearn_data/mrm_2010/correct_headers/C57*.nii.gz'))
    anat_files.remove(os.path.expanduser(
        '~/nilearn_data/mrm_2010/correct_headers/C57_ab1_invivo_corrected.nii.gz'))

    # Generate random transforms
    for anat_file in anat_files:
        atlas_file = anat_file.replace('C57', 'Atlas')
        if 'ab2' in anat_file:
            raw_atlas_file = atlas_file.replace('ab2', 'Ab2')
        elif 'y81' in anat_file:
            raw_atlas_file = atlas_file.replace('Invivo', 'invivo')
        else:
            raw_atlas_file = atlas_file

        param_file = fname_presuffix(anat_file.replace('_corrected', ''),
                                      newpath=sammba_params_dir,
                                      suffix='_bil2_transform.1D',
                                      use_ext=False)
        sammba_anat_file = fname_presuffix(anat_file,
                                           newpath=sammba_transformed_dir,
                                           prefix='bil2_transfo_')
        sammba_atlas_file = fname_presuffix(atlas_file,
                                           newpath=sammba_transformed_dir,
                                           prefix='bil2_transfo_')
        print(anat_file)
        assert(os.path.isfile(raw_atlas_file))

        allineate = afni.Allineate().run
        copy_geom = fsl.CopyGeom().run

        if True:
            out_allineate = allineate(
                in_file=anat_file,
                master=anat_file,
                in_param_file=param_file,
                nwarp='bilinear',
                out_file=sammba_anat_file,
                environ={'AFNI_DECONFLICT':'OVERWRITE'}
                )
            out_copy_geom = copy_geom(dest_file=sammba_anat_file,
                                      in_file=anat_file)
        out_allineate = allineate(
            in_file=raw_atlas_file,
            master=raw_atlas_file,
            in_param_file=param_file,
            nwarp='bilinear',
            final_interpolation='nearestneighbour',
            out_file=sammba_atlas_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'}
            )
        out_copy_geom = copy_geom(dest_file=sammba_atlas_file,
                                  in_file=raw_atlas_file)
