"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import glob
import numpy as np
from nilearn._utils.niimg_conversions import check_niimg
from nilearn import image
from sammba.externals.nipype.interfaces import afni, fsl
from sammba.externals.nipype.utils.filemanip import fname_presuffix
from skimage import measure


def mask_arrays_to_dice(mask_data1, mask_data2):
    numerator = np.logical_and(mask_data1, mask_data2).sum()
    denominator = mask_data1.sum() + mask_data2.sum()
    return 2 * numerator / float(denominator)

def compute_mask_contour(mask_file, write_dir=None, out_file=None):
    mask_img = check_niimg(mask_file)
    vertices, _, _, _ = measure.marching_cubes_lewiner(mask_img.get_data(), 0)  #marching_cubes_lewiner
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
            'Average_atlas_invivo.nii')
        spm_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented',
            'bil2_transformed')
    elif os.path.expanduser('~') == '/home/salma':
        spm_template_atlas_file = os.path.join(
            '/home/salma/appning_data/Pmamobipet/mrm_2010/reoriented',
            'Average_atlas_invivo.nii')
        spm_dir = os.path.join(
            '/home/salma/appning_data/Pmamobipet/mrm_2010/reoriented',
            'bil2_transformed')
    else:
        raise ValueError('Unknown user')

    sammba_dir = os.path.expanduser('~/mrm_reoriented_bil2_preprocessed')
    sammba_raw_dir = os.path.expanduser('~/mrm_reoriented_bil2')
    sammba_template_file = os.path.expanduser(
        '~/mrm_reoriented_bil2/Average_template_invivo.nii.gz')
    sammba_template_atlas_file = os.path.expanduser(
        '~/mrm_reoriented_bil2/Average_atlas_invivo.nii.gz')

    anat_files = glob.glob(os.path.expanduser(
        '~/mrm_reoriented_bil2/bil2_transfo_C57*.nii.gz'))
    mice_ids = [os.path.basename(a)[12:] for a in anat_files]
    sammba_dices = []
    sammba_dices2 = []
    spm_dices = []
    original_dices = []
    for mouse_id in mice_ids:
        atlas_id = mouse_id.replace('C57_', '')
        sammba_mouse_atlas_file = fname_presuffix(atlas_id,
                                                  newpath=sammba_raw_dir,
                                                  prefix='bil2_transfo_Atlas')
        matrix_file = fname_presuffix(mouse_id,
                                      newpath=sammba_dir,
                                      prefix='bil2_transfo',
                                      suffix='_unifized_masked_aff.aff12.1D',
                                      use_ext=False)
        warp_file = fname_presuffix(mouse_id,
                                    newpath=sammba_dir,
                                    prefix='bil2_transfo',
                                    suffix='_unifized_affine_general_warped_WARP')
        sammba_registered_atlas_file = fname_presuffix(atlas_id,
                                                       suffix='_allineated',
                                                       newpath=sammba_dir)
        allineate = afni.Allineate().run
        out_allineate = allineate(
            in_file=sammba_mouse_atlas_file,
            master=sammba_template_file,
            in_matrix=matrix_file,
            out_file=sammba_registered_atlas_file,
            final_interpolation='nearestneighbour',
            environ={'AFNI_DECONFLICT':'OVERWRITE'})

        copy_geom = fsl.CopyGeom().run
        out_copy_geom = copy_geom(dest_file=sammba_registered_atlas_file,
                                  in_file=sammba_template_atlas_file)

        if os.path.isfile(warp_file):
            nwarp_apply = afni.NwarpApply().run
            transforms = [warp_file, matrix_file]
            warp = "'"
            warp += ' '.join(transforms)
            warp += "'"
            out_warp_apply = nwarp_apply(in_file=sammba_mouse_atlas_file,
                                         master=sammba_template_file,
                                         warp=warp,
                                         interp='nearestneighbor',
                                         out_file=fname_presuffix(
                                                sammba_registered_atlas_file,
                                                suffix='_warped'))
        spm_atlas_id = atlas_id.replace('_corrected', '')
        if mouse_id == '_C57_ab2_invivo.nii.gz':
            spm_atlas_id = '_Ab2_invivo.nii.gz'
        elif mouse_id == '_C57_y81_Invivo.nii.gz':
            spm_atlas_id = '_y81_invivo.nii.gz'

        spm_registered_atlas_file = fname_presuffix(spm_atlas_id,
                                                    newpath=spm_dir,
                                                    prefix='wbil2_transfo_Atlas',
                                                    suffix='.nii',
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

        sammba_mouse_labels, sammba_mouse_dices = dices(
            sammba_template_atlas_file, sammba_registered_atlas_file)

#        sammba_mouse_labels2, sammba_mouse_dices2 = dices(
#            sammba_template_atlas_file, out_warp_apply.outputs.out_file)
        original_mouse_labels, original_mouse_dices = dices(
            sammba_template_atlas_file, sammba_mouse_atlas_file)
        spm_mouse_labels, spm_mouse_dices = dices(
            spm_template_atlas_file, uncropped_spm_registered_atlas_file)
        np.testing.assert_array_equal(sammba_mouse_labels, spm_mouse_labels)
#        np.testing.assert_array_equal(sammba_mouse_labels2, spm_mouse_labels)
        for (sammba_dice, spm_dice, label) in zip(sammba_mouse_dices, spm_mouse_dices,
                                                  sammba_mouse_labels):
            print('DICE for region {0} : spm {1}, sammba{2}'.format(label, spm_dice,
                 sammba_dice))
        sammba_dices.append(sammba_mouse_dices)
#        sammba_dices2.append(sammba_mouse_dices2)
        spm_dices.append(spm_mouse_dices)
        original_dices.append(original_mouse_dices)

    stop
    import matplotlib.pylab as plt


    print(np.argmin(np.abs(np.array(sammba_dices).T -
                           np.median(sammba_dices, axis=0)[:, np.newaxis]), axis=1))
    

    mask_imgs = [image.math_img('img=={}'.format(label), img=sammba_template_atlas_file)
                 for label in range(1, 20)]
    contour_imgs = [compute_mask_contour(mask_file, write_dir='/tmp',
                                         out_file=fname_presuffix(sammba_template_atlas_file,
                                                                  suffix='{}'.format(l))) for l, mask_file
                                         in enumerate(mask_imgs)]
    contour_img = image.math_img('np.sum(img, axis=-1)', img=contour_imgs)
    contour_img.to_filename('/tmp/contour.nii.gz')
    from nilearn import plotting

    anat_file = fname_presuffix(mice_ids[4],
                                newpath=sammba_dir,
                                prefix='bil2_transfo',
                                suffix='_unifized_affine_general')
    atlas_file = fname_presuffix(mice_ids[7].replace('C57_', ''),
                                newpath=sammba_dir,
                                suffix='_allineated')
    import shutil
    reoriented_registered_anat_file = '/tmp/mrm_registered.nii.gz'
    shutil.copy(anat_file, reoriented_registered_anat_file)
    out_copy_geom = copy_geom(dest_file=reoriented_registered_anat_file,
                              in_file=sammba_template_atlas_file)
    # -1.5 is good but shows mis accuracies
    ventricles_mask_img1 = image.math_img('img==9', img=sammba_template_atlas_file)
    ventricles_mask_img2 = image.math_img('img==10', img=sammba_template_atlas_file)
    ventricles_mask_img3 = image.math_img('img==8', img=sammba_template_atlas_file)
    ventricles_mask_img4 = image.math_img('img==14', img=sammba_template_atlas_file)
    ventricles_mask_img5 = image.math_img('img==16', img=sammba_template_atlas_file)
    display = plotting.plot_anat(reoriented_registered_anat_file, dim=-1.8, display_mode='z',
                                 cut_coords=[-2],
                                 annotate=False) #-2
    display.add_contours(ventricles_mask_img1, colors='r', linewidths=(.01, .01, .01, .01, .01, 1))  # 2, 3, 10
    display.add_contours(ventricles_mask_img2, colors='c', linewidths=(1, .01, .01, .01, .01, .01))  # 2, 3, 10
    display.add_contours(ventricles_mask_img3, colors='g', linewidths=(1, .01, .01, .01, .01, .01))  # 2, 3, 10
    display.add_contours(ventricles_mask_img4, colors='m', linewidths=(.01, .01, .01, .01, .01, 1))  # 2, 3, 10
    display.add_contours(ventricles_mask_img5, colors='b', linewidths=(1, .01, .01, .01, .01, .01))  # 2, 3, 10
    plt.savefig('/home/salma/publications/appning/poster/figures/atlas_overlays_dim-1pt8.png',
                facecolor='k', edgecolor='k')
    plotting.show()

    # 8 cerebellum, 1 hippocampus, 10 ventricles, 16 olfactory bulb
    # 4 and 20 are small
    # 6 internal capsule
    plt.style.use('dark_background')
    plt.figure(figsize=(3.5, 3.5))
    colors = ['r'] * 20
#    colors[8] = 'g'
#    colors[10] = 'b'
    colors[1] = 'c'
    colors[9] = 'y'
    labels = ['other regions'] * 20
#    labels[7] = 'cerebellum'
#    labels[0] = 'hippocampus'
    labels[1] = 'corpus callosum'
#    labels[4] = 'Anterior commissure'
    labels[9] = 'ventricles'
#    labels[16] = 'midbrain'
    for spm_region_dices, sammba_region_dices, color, label in zip(np.array(spm_dices).T,
                                                                   np.array(sammba_dices).T,
                                                                   colors, labels):
        if color != 'r':
            plt.scatter(spm_region_dices, sammba_region_dices, c=color, label=label, s=5)
        else:
            plt.scatter(spm_region_dices, sammba_region_dices, c=color, s=5)

    plt.scatter(spm_region_dices, sammba_region_dices, c=color, label=label, s=5)
    plt.plot([0, 1], [0, 1], 'w')
    plt.legend(fontsize=13)
    m = max(np.max(sammba_dices), np.max(spm_dices))
    plt.xlim(.2 - .01, m + .01)
    plt.ylim(.2- .01, m + .01)
    ticks = [.3, .5, .7, .9]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('SPM mouse', fontsize=13)
    plt.ylabel('sammba-MRI', fontsize=13)
    plt.subplots_adjust(top=.97, right=.98, bottom=.16, left=.17)
    plt.savefig(os.path.expanduser('~/publications/appning/poster/figures/bil2_transfo_dice_boxplots.png'))
    plt.show()