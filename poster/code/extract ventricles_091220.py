import nibabel
from nilearn import image
from scipy import ndimage



dorr_img = nibabel.load('/home/salma/nilearn_data/dorr_2008/Dorr_2008_labels_100um.nii.gz')
right_img = image.math_img('img==57', img=dorr_img)
left_img = image.math_img('img==77', img=dorr_img)
third_img = image.math_img('img==146', img=dorr_img)
fourth_img = image.math_img('img==118', img=dorr_img)
image.concat_imgs([right_img, left_img, third_img, fourth_img]).to_filename('/tmp/dorr_masked.nii.gz')


img = nibabel.load('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf/mouse_091220/reoriented/anat_n0_clear_hd.nii')
mask_img = image.math_img('img > 18500', img=img)
labeling = ndimage.label(mask_img.get_data()[..., 0])
image.new_img_like(img, labeling[0]).to_filename('/tmp/perf_labeled.nii.gz')
right = labeling[0] == 627  #57 OK
third_left = labeling[0] == 46  # 146 (not good)
fourth = labeling[0] == 570  # 118 OK

mask_img = image.math_img('img > 19000', img=img)
opened_data = ndimage.binary_opening(mask_img.get_data(), iterations=3)
labeling = ndimage.label(opened_data)
labeling = ndimage.label(mask_img.get_data())
image.new_img_like(img, labeling[0]).to_filename('/tmp/perf_labeled3.nii.gz')
                   
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



unbiased_img = nibabel.load('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf/mouse_091220/reoriented/anat_n0_clear_hd_n4.nii')
mask_img = image.math_img('img > 10000', img=unbiased_img)
opened_data = ndimage.binary_opening(mask_img.get_data())
labeling = ndimage.label(opened_data)
image.new_img_like(img, labeling[0]).to_filename('/tmp/perf_labeled3.nii.gz')
right = labeling[0] == 155  #57
left = labeling[0] == 63  # 77 OK
third = labeling[0] == 137  # 146 (not good)
fourth = labeling[0] == 46  # 118

mask_img = image.math_img('img > 11000', img=unbiased_img)
labeling = ndimage.label(mask_img.get_data())
image.new_img_like(img, labeling[0]).to_filename('/tmp/perf_labeled4.nii.gz')

dirty_three_ventricles = labeling[0] == 824  #57
three_ventricles = dirty_three_ventricles * third_left
image.new_img_like(img, three_ventricles).to_filename('/tmp/three.nii.gz')

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
               img2='/tmp/fourth_mask.nii.gz').to_filename('/home/salma/appning_data/Pmamobipet/inhouse_mouse_perf/mouse_091220/reoriented/ventricles_mask.nii')
