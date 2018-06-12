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


def dice(labels_file1, mask_file2):
    mask_data1 = check_niimg(mask_file1).get_data() > 0
    mask_data2 = check_niimg(mask_file2).get_data() > 0
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

if __name__ == '__main__':
    if os.path.expanduser('~') == '/home/bougacha':
        spm_template_atlas_file = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented',
            'Average_atlas_invivo.img')
        spm_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented')
        sammba_template_file = os.path.join(
            '/home/bougacha/nilearn_data/mrm_2010',
            'Average_template_invivo.nii.gz')
        sammba_template_atlas_file = os.path.join(
            '/home/bougacha/nilearn_data/mrm_2010',
            'Average_atlas_invivo.nii.gz')
    elif os.path.expanduser('~') == '/home/salma':
        spm_dir = os.path.join('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf',
                               'mouse_091220/reoriented')
        sammba_dir = os.path.join('/home/salma/appning_data/my_home',
                                  'inhouse_mouse_perf_to_reoriented_head100',
                                  'mouse_091220')
    else:
        raise ValueError('Unknown user')

    sammba_dir = os.path.expanduser('~/mrm_preprocessed')
    sammba_raw_dir = os.path.expanduser('~/nilearn_data/mrm_2010')

    anat_files = glob.glob(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57*.nii.gz'))
    anat_files.remove(os.path.expanduser(
        '~/nilearn_data/mrm_2010/C57_Az1_invivo.nii.gz'))
    mice_ids = [os.path.basename(a)[3:] for a in anat_files]
    for mouse_id in mice_ids:
        sammba_mouse_atlas_file = fname_presuffix(mouse_id,
                                                  newpath=sammba_raw_dir,
                                                  prefix='Atlas')
        matrix_file = fname_presuffix(mouse_id,
                                      newpath=sammba_raw_dir,
                                      prefix='C57',
                                      suffix='_unifized_masked_aff.aff12.1D',
                                      useext=False)
        sammba_registered_atlas_file = fname_presuffix(mouse_id,
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
    
        spm_mouse_atlas_file = fname_presuffix(mouse_id,
                                               newpath=spm_dir,
                                               prefix='Atlas')
        spm_registered_atlas_file = fname_presuffix(mouse_id,
                                                    newpath=spm_dir,
                                                    prefix='w')
    
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
            uncropped_spm_registered_atlasr_file = spm_registered_atlas_file

        sammba_dice = dice(sammba_template_atlas_file,
                           sammba_registered_atlas_file)
        spm_dice = dice(spm_template_atlas_file,
                        uncropped_spm_registered_atlas_file)
        sammba_dices = []
        spm_dices = []
        for (dice_coef, label) in zip([sammba_dice, spm_dice],
                                      ['sammba', 'spm']):
            print('ventricles DICE for ' + label, dice_coef)
        sammba_dices.append(sammba_dice)
        spm_dices.append(spm_dice)

    import matplotlib.pylab as plt

    plt.boxplot(sammba_dices, positions=[1])
    plt.boxplot(spm_dices, positions=[2])
    plt.xlim([0.5, 2.5])
    plt.xticks([1, 2], ['sammba', 'spmmouse'])
    plt.ylabel('dice coefficient')
    plt.savefig(os.path.expanduser('~/publications/appning/poster/figures/dice_boxplots.png'))
    plt.show()