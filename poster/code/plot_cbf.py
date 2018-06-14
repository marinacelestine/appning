from nilearn import plotting
from nilearn import image
import matplotlib.pylab as plt

mean_cbf = '/home/bougacha/test_design/cbfs/mean_cbf_brain.nii.gz'
img = image.smooth_img(mean_cbf, fwhm=[0.2, 0.2, 0.2])
smoothed_cbf_img = image.math_img('(img1 !=0) * img2', img1=mean_cbf, img2=img)
smoothed_cbf_img.to_filename(
    '/home/bougacha/test_design/cbfs/smoothed_mean_cbf_brain.nii.gz')

plotting.plot_stat_map(smoothed_cbf_img,
                       bg_img='/home/bougacha/nilearn_data/dorr_2008/Dorr_2008_average.nii.gz',
                       vmax=220, display_mode='y', cut_coords=[1.1], annotate=False,
                       cmap=plt.get_cmap('jet'),
                       )
plt.savefig('/home/bougacha/papers/appning/poster/figures/smoothed_mean_cbf_y_no_title.jpg', facecolor='k', edgecolor='k')
#plt.savefig('/home/bougacha/papers/appning/Images/mean_cbf_y.jpg', facecolor='k', edgecolor='k')
#plt.savefig('/home/bougacha/papers/appning/Images/mean_cbf_y.tif', facecolor='k', edgecolor='k')
plotting.show()

from nilearn.input_data import NiftiLabelsMasker
import glob
from sammba import data_fetchers

cbfs = glob.glob('/home/bougacha/inhouse_mouse_perf_preprocessed_mine/*/perfFAIREPI_n0_cbf_perslice_to_head100.nii.gz')
atlas = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/labels100.nii.gz'
brain_mask = '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/mask100.nii.gz'
from nilearn import image, masking
mean_cbf = image.mean_img(cbfs)
plausible_cbf_mask = image.math_img('np.logical_and(img> 0, img < 250)',
                                    img=mean_cbf)
mask_img = masking.intersect_masks([plausible_cbf_mask, brain_mask],
                                   threshold=1)
dorr_masker = NiftiLabelsMasker(atlas, resampling_target=None,
                                mask_img=mask_img, detrend=False,
                                standardize=False)
roi_cbfs = dorr_masker.fit_transform(cbfs)


import matplotlib.pylab as plt
import numpy as np

dorr = data_fetchers.fetch_atlas_dorr_2008()
hippocampus_names = [u'R hippocampus',
                     u'L hippocampus',
                     u'R stratum granulosum of hippocampus',
                     u'L stratum granulosum of hippocampus',
                     u'L dentate gyrus of hippocampus',
                     u'R dentate gyrus of hippocampus']
hippocampal_indices = [n for n, l in enumerate(dorr.names)
                       if l in hippocampus_names]
hippocampal_indices = [10, 32, 12, 34]  # reorder
hippocampal_indices = [12, 34, 3, 46]  # reorder

plt.style.use('dark_background')
plt.figure(figsize=(4, 3))
plt.boxplot(roi_cbfs[:, hippocampal_indices], vert=False,
            boxprops={'color':'y', 'linewidth':2},
            medianprops={'color':'m', 'linewidth':2},
            whiskerprops={'color':'y', 'linewidth':2})
labels = [r.replace(' of', '\nof') for  r in dorr.names[hippocampal_indices]]
plt.yticks(1 + np.arange(len(hippocampal_indices)),
           labels, fontsize=13)
plt.subplots_adjust(top=.98, right=.98, bottom=.08, left=.4)
plt.xticks([80, 120, 160, 200])
plt.savefig('/home/bougacha/papers/appning/poster/figures/regional_cbf_4rois_black_bg.jpg', facecolor='k', edgecolor='k')
plt.show()