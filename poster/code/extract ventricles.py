import nibabel
from nilearn import image
from scipy import ndimage



dorr_img = nibabel.load('/home/salma/appning_data/Pmamobipet/templates/mouse/reoriented/labels100.nii')
right_img = image.math_img('img==57', img=dorr_img)
left_img = image.math_img('img==77', img=dorr_img)
third_img = image.math_img('img==146', img=dorr_img)
fourth_img = image.math_img('img==118', img=dorr_img)
img = image.math_img('np.sum(imgs, axis=-1).astype(float)',
                     imgs=[right_img, left_img, third_img, fourth_img])
img.to_filename(
    '/home/salma/appning_data/my_home/inhouse_mouse_perf_to_reoriented_head100/labels100_ventricles_mask.nii.gz')


img = nibabel.load('/home/salma/nilearn_data/zurich_retest/baseline/1366/3DRARE.nii.gz')
mask_img = image.math_img('img > .35', img=img)
opened_data = ndimage.binary_opening(mask_img.get_data())
labeling = ndimage.label(opened_data)
right = labeling[0] == 18  #57
left = labeling[0] == 50  # 77
third = labeling[0] == 43  # 146 (not good)
fourth = labeling[0] == 42  # 118

# Dilate extracted objects and mask the initial image
dilated_ventricles = [ndimage.binary_dilation(v_data, iterations=1)
                      for v_data in [right, left, third, fourth]]
masked_imgs = [image.math_img('img1 * img2', img1=img,
                              img2=image.new_img_like(img, dilated_v_data))
               for dilated_v_data in dilated_ventricles]
image.concat_imgs(masked_imgs).to_filename('/tmp/masked.nii.gz')
labeling = ndimage.label(mask_img.get_data())
fourth = labeling[0] == 1599
image.new_img_like(img, fourth).to_filename('/tmp/fourth_mask.nii.gz')



unbiased_img = nibabel.load('/home/salma/nilearn_data/zurich_retest/baseline/1366/3DRARE_unbiased.nii.gz')
mask_img = image.math_img('img > .18', img=unbiased_img)
opened_data = ndimage.binary_opening(mask_img.get_data())
labeling = ndimage.label(opened_data)
three_ventricles = labeling[0] == 65  #57
fourth = labeling[0] == 163  # 118
masked_img = image.math_img('img1 * img2', img1=img,
                              img2=image.new_img_like(img, three_ventricles))
masked_img.to_filename('/tmp/three.nii.gz')
new_mask_img = image.math_img('img>.25', img=masked_img)
labeling = ndimage.label(new_mask_img.get_data())
image.new_img_like(new_mask_img, labeling[0]).to_filename('/tmp/labeling2.nii.gz')
final_three = ndimage.binary_fill_holes(labeling[0] == 1)
image.new_img_like(img, final_three).to_filename('/tmp/ventricles3_mask.nii.gz')
image.math_img('img1 * (img2==1)', img1=img,
               img2='/tmp/labeling2.nii.gz').to_filename('/tmp/three2.nii.gz')


image.math_img('img1 + img2', img1='/tmp/ventricles3_mask.nii.gz',
               img2='/tmp/fourth_mask.nii.gz').to_filename('/home/salma/nilearn_data/zurich_retest/baseline/1366/ventricles_mask.nii.gz')
