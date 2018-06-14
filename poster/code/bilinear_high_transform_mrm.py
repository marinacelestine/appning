"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import glob
import numpy as np
import nibabel
from sammba.externals.nipype.interfaces import afni, fsl
from sammba.externals.nipype.utils.filemanip import fname_presuffix


if __name__ == '__main__':
    if os.path.expanduser('~') == '/home/bougacha':
        spm_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented')
        spm_transformed_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented',
            'bil2_transformed')
    elif os.path.expanduser('~') == '/home/salma':
        spm_dir = os.path.join('/home/salma/appning_data/Pmamobipet/mrm_2010',
                               'reoriented', 'bil2_transformed')
    else:
        raise ValueError('Unknown user')

    sammba_dir = os.path.expanduser('~/mrm_preprocessed')
    sammba_raw_dir = os.path.expanduser('~/nilearn_data/mrm_2010')
    sammba_transformed_dir = os.path.expanduser('~/mrm_bil2_transformed')

    anat_files = glob.glob(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57*.nii.gz'))
    anat_files.remove(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57_ab1_invivo.nii.gz'))

    # Generate random transforms
    rand_gen = np.random.RandomState(12)
    idenity_transform = np.zeros((43,))
    idenity_transform[6:9] = np.ones((3,))
    std = .005 * np.ones((43,))  # 0.025
    std[:6] = 0.1
    std[6:9] = 0.02
    std[9:12] = 0.01
    transforms = idenity_transform + rand_gen.randn(len(anat_files), 43) * std
    for anat_file, transform in zip(anat_files, transforms):
        atlas_file = anat_file.replace('C57', 'Atlas')
        reoriented_anat_file = fname_presuffix(anat_file,
                                               newpath=spm_dir,
                                               suffix='.nii',
                                               use_ext=False)
        if 'ab2' in anat_file:
            raw_atlas_file = atlas_file.replace('ab2', 'Ab2')
        elif 'y81' in anat_file:
            raw_atlas_file = atlas_file.replace('Invivo', 'invivo')
        else:
            raw_atlas_file = atlas_file

        reoriented_atlas_file = fname_presuffix(raw_atlas_file,
                                                newpath=spm_dir,
                                                suffix='.nii',
                                                use_ext=False)
        print(reoriented_anat_file, reoriented_atlas_file)
        param_file = fname_presuffix(anat_file,
                                      newpath=sammba_transformed_dir,
                                      suffix='_bil2_transform.1D',
                                      use_ext=False)
        sammba_anat_file = fname_presuffix(anat_file,
                                           newpath=sammba_transformed_dir,
                                           prefix='bil2_transfo_')
        sammba_atlas_file = fname_presuffix(atlas_file,
                                           newpath=sammba_transformed_dir,
                                           prefix='bil2_transfo_')
        spm_anat_file = fname_presuffix(anat_file,
                                        newpath=spm_transformed_dir,
                                        prefix='bil2_transfo_',
                                        suffix='.nii',
                                        use_ext=False)
        spm_atlas_file = fname_presuffix(atlas_file,
                                        newpath=spm_transformed_dir,
                                        prefix='bil2_transfo_',
                                        suffix='.nii',
                                        use_ext=False)
        print(anat_file)
        assert(os.path.isfile(raw_atlas_file))
#        if os.path.isfile(param_file):
        np.savetxt(param_file, transform, newline='    ',
                   fmt='%10.10f')

        allineate = afni.Allineate().run
        out_allineate = allineate(
            in_file=anat_file,
            master=anat_file,
            in_param_file=param_file,
            nwarp='bilinear',
            out_file=sammba_anat_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
        copy_geom = fsl.CopyGeom().run
        out_copy_geom = copy_geom(dest_file=sammba_anat_file,
                                  in_file=anat_file)
        out_allineate = allineate(
            in_file=reoriented_anat_file,
            master=reoriented_anat_file,
            in_param_file=param_file,
            nwarp='bilinear',
            out_file=spm_anat_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
        out_copy_geom = copy_geom(dest_file=spm_anat_file,
                                  in_file=reoriented_anat_file)
        out_allineate = allineate(
            in_file=raw_atlas_file,
            master=raw_atlas_file,
            in_param_file=param_file,
            nwarp='bilinear',
            final_interpolation='nearestneighbour',
            out_file=sammba_atlas_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
        out_copy_geom = copy_geom(dest_file=sammba_atlas_file,
                                  in_file=raw_atlas_file)
        out_allineate = allineate(
            in_file=reoriented_atlas_file,
            master=reoriented_atlas_file,
            in_param_file=param_file,
            nwarp='bilinear',
            final_interpolation='nearestneighbour',
            out_file=spm_atlas_file,
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
        out_copy_geom = copy_geom(dest_file=spm_atlas_file,
                                  in_file=reoriented_atlas_file)