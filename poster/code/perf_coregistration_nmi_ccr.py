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


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


if __name__ == '__main__':
    mouse_dir = 'mouse_091220'
    mouse_dir = 'mouse_191851'
    my_dir = os.path.join(
        os.path.expanduser('~/inhouse_mouse_perf_preprocessed_mine'),
        mouse_dir)
    if os.path.expanduser('~') == '/home/bougacha':
        spm_dir = os.path.join(
            '/home/Pmamobipet/0_Dossiers-Personnes',
            'Salma/inhouse_mouse_perf_preprocessed_mine',
            mouse_dir)
    elif os.path.expanduser('~') == '/home/salma':
        spm_dir = os.path.join('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf',
                               mouse_dir, 'reoriented')
    else:
        raise ValueError('Unknown user')

    my_perf1 = os.path.join(
        my_dir,
        'perfFAIREPI_n0_M0_unbiased_perslice_oblique.nii.gz')
    spm_anat = os.path.join(
        spm_dir,
        'anat_n0_clear_hd.nii')  # anat_n0_unifized_clean_hd.nii
    spm_perf = os.path.join(
        spm_dir,
        'rperfFAIREPI_n0_M0.nii') # replace by rperfFAIREPI_n0_M0_unbiased_clean_hd.nii)


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
    my_perf_in_anat_space = out_allineate.outputs.out_file
    my_perf_data = nibabel.load(my_perf_in_anat_space).get_data()
    my_perf_mask_img = image.math_img('img > {}'.format(my_perf_data.min()),
                                      img=my_perf_in_anat_space)
    my_perf_mask_data = my_perf_mask_img.get_data().astype(bool)
    my_perf_mask = fname_presuffix(my_perf_in_anat_space, suffix='_mask')
    my_perf_mask_img.to_filename(my_perf_mask)


    spm_perf_data = nibabel.load(spm_perf).get_data()
    spm_perf_mask_img = image.math_img(
        'img > {}'.format(np.nanmin(spm_perf_data)),
        img=my_perf_in_anat_space)
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

    l_out = l(in_file1=my_anat, in_file2=my_perf_in_anat_space,
              mask_file=my_perf_mask,
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

    import matplotlib.pylab as plt

    data = [my_cr_data[my_perf_mask_data].ravel(),
            spm_cr_data[spm_perf_mask_data].ravel()]
    parts = plt.violinplot(
            data, showmeans=False, showmedians=False,
            showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    sammba_quartile1, sammba_median, sammba_quartile3 = np.percentile(data[0], [25, 50, 75])
    spm_quartile1, spm_median, spm_quartile3 = np.percentile(data[1], [25, 50, 75])
    quartile1 = [sammba_quartile1, spm_quartile1]
    medians = [sammba_median, spm_median]
    quartile3 = [sammba_quartile3, spm_quartile3] 
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    # set style for the axes
    labels = ['sammba', 'spmmouse']
    set_axis_style(plt.gca(), labels)
    
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()