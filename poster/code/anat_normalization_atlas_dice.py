"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import glob
import numpy as np
from skimage import measure
from nilearn import image, masking
from nilearn._utils.niimg_conversions import check_niimg
from sammba.externals.nipype.interfaces import afni
from sammba.externals.nipype.caching import Memory
from sammba.externals.nipype.utils.filemanip import fname_presuffix


def mask_arrays_to_dice(mask_data1, mask_data2):
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

    
def dices(labels_file1, labels_file2):
    data1 = check_niimg(labels_file1).get_data().astype(int)
    data2 = check_niimg(labels_file2).get_data().astype(int)
    labels1 = np.unique(data1).tolist()
    labels2 = np.unique(data2).tolist()
    dice_coefs = []
    labels = np.unique(labels1 + labels2).tolist()
    labels.remove(0)
    for label in labels:
        if label in labels1 and label in labels2:
            mask_data1 = data1 == label
            mask_data2 = data2 == label
            dice_coefs.append(mask_arrays_to_dice(mask_data1, mask_data2))
        else:
            dice_coefs.append(0.)
            
    return labels, dice_coefs

if __name__ == '__main__':
    if os.path.expanduser('~') == '/home/bougacha':
        spm_template_atlas_file = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented',
            'Average_atlas_invivo.img')
        spm_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented')
    elif os.path.expanduser('~') == '/home/salma':
        spm_template_atlas_file = os.path.join(
            '/home/salma/appning_data/Pmamobipet/mrm_2010/reoriented',
            'Average_atlas_invivo.img')
        spm_dir = '/home/salma/appning_data/Pmamobipet/mrm_2010/reoriented'
        sammba_dir = '/home/salma/mrm_preprocessed'
    else:
        raise ValueError('Unknown user')

    sammba_dir = os.path.expanduser('~/mrm_preprocessed')
    sammba_raw_dir = os.path.expanduser('~/nilearn_data/mrm_2010')
    sammba_template_file = os.path.join(sammba_raw_dir,
                                        'Average_template_invivo.nii.gz')
    sammba_template_atlas_file = os.path.join(sammba_raw_dir,
                                              'Average_atlas_invivo.nii.gz')

    anat_files = glob.glob(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57*.nii.gz'))
    anat_files.remove(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57_Az1_invivo.nii.gz'))
    anat_files.remove(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57_ab1_invivo.nii.gz'))
    mice_ids = [os.path.basename(a)[3:] for a in anat_files]
    sammba_dices = []
    spm_dices = []
    original_dices = []
    for mouse_id in mice_ids:
        if mouse_id == '_ab2_invivo.nii.gz':
            atlas_id = '_Ab2_invivo.nii.gz'
        elif mouse_id == '_y81_Invivo.nii.gz':
            atlas_id = '_y81_invivo.nii.gz'
        else:
            atlas_id = mouse_id
            
        sammba_mouse_atlas_file = fname_presuffix(atlas_id,
                                                  newpath=sammba_raw_dir,
                                                  prefix='Atlas')
        matrix_file = fname_presuffix(mouse_id,
                                      newpath=sammba_dir,
                                      prefix='C57',
                                      suffix='_unifized_masked_aff.aff12.1D',
                                      use_ext=False)
        sammba_registered_atlas_file = fname_presuffix(atlas_id,
                                                       suffix='_allineated',
                                                       newpath=sammba_dir)
        allineate = afni.Allineate().run
        out_allineate = allineate(
            in_file=sammba_mouse_atlas_file,
            master=sammba_template_file,
            in_matrix=matrix_file,
            out_file=sammba_registered_atlas_file,
            interpolation='nearestneighbour',
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
    
        spm_registered_atlas_file = fname_presuffix(mouse_id,
                                                    newpath=spm_dir,
                                                    prefix='wAtlas',
                                                    suffix='.img',
                                                    use_ext=False)
    
        spm_registered_atlas_data = check_niimg(spm_registered_atlas_file).get_data()
        template_shape = check_niimg(spm_template_atlas_file).shape
        if spm_registered_atlas_data.shape != template_shape:
            uncropped_spm_registered_atlas_data = np.vstack(
                (np.zeros((1,) + template_shape[1:]),
                 spm_registered_atlas_data))
            uncropped_spm_registered_atlas_data = np.nan_to_num(
                uncropped_spm_registered_atlas_data)
            uncropped_spm_registered_atlas_file = fname_presuffix(
                spm_registered_atlas_file, suffix='_uncropped')
            image.new_img_like(
                spm_registered_atlas_file,
                uncropped_spm_registered_atlas_data).to_filename(
                    uncropped_spm_registered_atlas_file)
                
        else:
            uncropped_spm_registered_atlas_file = spm_registered_atlas_file

        sammba_mouse_labels, sammba_mouse_dices = dices(sammba_template_atlas_file,
                                                        sammba_registered_atlas_file)

        original_mouse_labels, original_mouse_dices = dices(sammba_template_atlas_file,
                                                            sammba_mouse_atlas_file)

        spm_mouse_labels, spm_mouse_dices = dices(spm_template_atlas_file,
                                                  uncropped_spm_registered_atlas_file)
        np.testing.assert_array_equal(sammba_mouse_labels, spm_mouse_labels)
        for (sammba_dice, spm_dice, label) in zip(sammba_mouse_dices, spm_mouse_dices,
                                                  sammba_mouse_labels):
            print('DICE for region {0} : spm {1}, sammba{2}'.format(label, spm_dice,
                 sammba_dice))
        sammba_dices.append(sammba_mouse_dices)
        spm_dices.append(spm_mouse_dices)
        original_dices.append(original_mouse_dices)

    import matplotlib.pylab as plt

    plt.boxplot(sammba_dices, positions=range(10))
    plt.boxplot(spm_dices, positions=np.arange(10) + .15)
    plt.boxplot(original_dices, positions=range(10) + .3)

    plt.xticks([1, 2], ['sammba', 'spmmouse'])
    plt.ylabel('dice coefficient')
    plt.savefig(os.path.expanduser('~/publications/appning/poster/figures/dice_boxplots.png'))
    plt.show()