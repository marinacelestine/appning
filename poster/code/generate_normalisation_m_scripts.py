import os
import glob
from sammba.externals.nipype.utils.filemanip import fname_presuffix

anat_files = glob.glob(
    '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented/transformed/transfo_C57*.nii')

common_m_file = '/home/Pmamobipet/0_Dossiers-Personnes/Salma/batches/c57_mrm_normalization_common.m'
with open(common_m_file, 'r') as file_content:
    common_lines = file_content.read()

for anat_file in anat_files:
    base_file = os.path.basename(anat_file)
    atlas_base_file = base_file.replace('C57', 'Atlas')

    if 'ab2' in anat_file:
        atlas_base_file = atlas_base_file.replace('ab2', 'Ab2')
    elif 'y81' in anat_file:
        atlas_base_file = atlas_base_file.replace('Invivo', 'invivo')

    m_script_file = fname_presuffix(anat_file[:-11],
                                    newpath='/home/Pmamobipet/0_Dossiers-Personnes/Salma/batches',
                                    prefix='normalize_', suffix='.m',
                                    use_ext=False)
    print(m_script_file)
    windows_atlas_file = "\'Y:\\0_Dossiers-Personnes\\Salma\\mrm_2010\\reoriented\\transformed\\" +\
        atlas_base_file + ",1\'"

    windows_transform_file = "\'Y:\\0_Dossiers-Personnes\\Salma\\mrm_2010\\reoriented\\transformed\\" +\
        base_file[:-4] + "_seg_sn.mat\'"

    lines = "matlabbatch{1}.spm.spatial.normalise.write.subj.matname = {"
    lines += windows_transform_file
    lines += '};\n'

    lines += "matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {"
    lines += windows_atlas_file
    lines += '};\n'

    lines += common_lines
    with open(m_script_file, 'w') as m_script_lines:
        m_script_lines.write(lines)
        

