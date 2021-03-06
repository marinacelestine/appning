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
    sammba_dices = []
    spm_dices = []
    for mouse_id in ['mouse_091220', 'mouse_102617']:
        if os.path.expanduser('~') == '/home/bougacha':
            template_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma',
                                        'templates/mouse/reoriented')
            spm_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes',
                                   'Salma/inhouse_mouse_perf', mouse_id,
                                   'reoriented')
            sammba_dir = os.path.join('/home/bougacha',
                                      'inhouse_mouse_perf_to_reoriented_head100',
                                      mouse_id)
        elif os.path.expanduser('~') == '/home/salma':
            template_dir = os.path.join('/home/salma/appning_data/Pmamobipet',
                                        'templates/mouse/reoriented')
            spm_dir = os.path.join('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf',
                                   mouse_id, 'reoriented')
            sammba_dir = os.path.join('/home/salma/appning_data/my_home',
                                      'inhouse_mouse_perf_to_reoriented_head100',
                                      mouse_id)
            sammba_dir = os.path.join(
                os.path.expanduser('~/inhouse_mouse_perf_to_reoriented_head100'), mouse_id)
        else:
            raise ValueError('Unknown user')
    
        template_ventricles_mask_file = os.path.join(os.path.expanduser(
             '~/appning_data/my_home/inhouse_mouse_perf_to_reoriented_head100/labels100_ventricles_mask.nii.gz'))
    
        # Create a precise brain mask, by combining RATS and SPM
        ventricles_mask_file = os.path.join(spm_dir,
                                            'manual_ventricles_mask.nii')
    
        sammba_registered_ventricles_file = fname_presuffix(ventricles_mask_file,
                                                            suffix='_allineated',
                                                            newpath=sammba_dir)
        allineate = afni.Allineate().run
        out_allineate = allineate(
            in_file=ventricles_mask_file,
            master=template_ventricles_mask_file,
            in_matrix=os.path.join(sammba_dir, 'anat_n0_unifized_masked_aff.aff12.1D'),
            out_file=sammba_registered_ventricles_file,
            interpolation='nearestneighbour',
            environ={'AFNI_DECONFLICT':'OVERWRITE'})
    
        spm_registered_ventricles_file = os.path.join(
            spm_dir, 'wmanual_ventricles_mask.nii')
    
        spm_registered_ventricles_data = check_niimg(spm_registered_ventricles_file).get_data()
        template_shape = check_niimg(template_ventricles_mask_file).shape
        if spm_registered_ventricles_data.shape != template_shape:
            uncropped_spm_registered_ventricles_data = np.vstack(
                (np.zeros((1,) + template_shape[1:]), spm_registered_ventricles_data))
            uncropped_spm_registered_ventricles_data = np.nan_to_num(
                uncropped_spm_registered_ventricles_data)
            uncropped_spm_registered_contour_file = fname_presuffix(
                spm_registered_ventricles_file, suffix='_uncropped')
            image.new_img_like(
                spm_registered_ventricles_file,
                uncropped_spm_registered_ventricles_data).to_filename(
                    uncropped_spm_registered_contour_file)
                
        else:
            uncropped_spm_registered_contour_file = spm_registered_ventricles_file

        sammba_dice = dice(template_ventricles_mask_file, sammba_registered_ventricles_file)
        spm_dice = dice(template_ventricles_mask_file, uncropped_spm_registered_contour_file)
        for (dice_coef, label) in zip([sammba_dice, spm_dice], ['sammba', 'spm']):
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