"""
This code labels each voxel from normalized anatomical to corresponding tissue
and computes Dice coefficient between registeraed and template images
for each tissue
"""
import os
import numpy as np
from nilearn import image
from nilearn._utils.niimg_conversions import check_niimg
from sklearn.metrics import mutual_info_score
from sammba.externals.nipype.interfaces.afni import utils

def compute_nmi(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi = mi / max(np.sqrt(h_true * h_pred), 1e-10)
    return nmi

if __name__ == '__main__':
    template_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes/Salma',
                                'inhouse_mouse_perf/template/with_mask')
    registered_dir = os.path.join('/home/Pmamobipet/0_Dossiers-Personnes',
                                  'Salma/inhouse_mouse_perf_preprocessed_mine',
                                  'mouse_191851/reoriented')

    template_tissues_imgs = [
        os.path.join(template_dir, 'c{0}head100.nii'.format(n))
        for n in range(1, 4)]
    template_mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .000001', imgs=template_tissues_imgs)
    template_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=template_mask_img,
        imgs=template_tissues_imgs)

    registered_tissues_imgs = [
        os.path.join(
            registered_dir,
            'c{0}anat_n0_unifized_affine_general_warped_clean_hd.nii'.format(n))
        for n in range(1, 4)]
    registered_mask_img = image.math_img(
        'np.max(imgs, axis=-1) > .000001', imgs=registered_tissues_imgs)
    registered_img = image.math_img(
        'img * (np.argmax(imgs, axis=-1) + 1)',
        img=registered_mask_img,
        imgs=registered_tissues_imgs)

    for label in [1, 2, 3]:
        tissue_template_img = image.math_img('img=={0}'.format(label),
                                             img=template_img)
        tissue_registered_img = image.math_img('img=={0}'.format(label),
                                               img=registered_img)
    print(dice(tissue_template_img, tissue_registered_img))

    l = utils.LocalBistat()
    l.inputs.neighborhood = ('RECT', (-2, -2, -2))
    l.inputs.in_file2 =  '/home/Promane/2014-ROMANE/5_Experimental-Plan-Experiments-Results/mouse/MRIatlases/MIC40C57/20170223/brain100.nii.gz'

    l.inputs.stat = ['crU', 'spearman', 'normuti', 'mutinfo']

    l.inputs.automask = True

    preprocessing_dir = '/home/bougacha/inhouse_mouse_perf_preprocessed_mine'
    for mouse_dir in ['mouse_191851']:
        l.inputs.in_file1 = os.path.join(
            preprocessing_dir, mouse_dir,
            'anat_n0_unifized_affine_general_warped.nii.gz')
        l_out = l.run()
        cor_img = image.index_img(l_out.outputs.out_file, 0)
        nmi_img = image.index_img(l_out.outputs.out_file, 1)
        
        jip_dir = '/home/bougacha/jip_data'
        l.inputs.in_file1 = os.path.join(jip_dir, 'anat_191851_in_head_100.nii')
        
        l_out = l.run()
        jip_cor_img = image.index_img(l_out.outputs.out_file, 0)
        jip_nmi_img = image.index_img(l_out.outputs.out_file, 1)

        assert_less_equal(np.sum(nmi_img.get_data() > jip_nmi_img.get_data()),
                          np.sum(nmi_img.get_data() < jip_nmi_img.get_data()))
        
        assert_less(nmi_img.get_data().mean(), jip_nmi_img.get_data().mean())
        assert_less(jip_cor_img.get_data().mean(), cor_img.get_data().mean())

