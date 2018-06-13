import os
import glob
from sammba.externals.nipype.utils.filemanip import fname_presuffix

anat_files = glob.glob(
    '/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented/transformed/transfo_C57*.nii')

common_m_file = '/home/Pmamobipet/0_Dossiers-Personnes/Salma/batches/c57_mrm_segmentation_common.m'
with open(common_m_file, 'r') as file_content:
    common_lines = file_content.read()

for anat_file in anat_files:
    base_file = os.path.basename(anat_file)
    m_script_file = fname_presuffix(anat_file[:-11],
                                    newpath='/home/Pmamobipet/0_Dossiers-Personnes/Salma/batches',
                                    prefix='segment_', suffix='.m',
                                    use_ext=False)
    print(m_script_file)
    windows_anat_file = "\'Y:\\0_Dossiers-Personnes\\Salma\\mrm_2010\\reoriented\\transformed\\" +\
        base_file + ",1\'"

    lines = "matlabbatch{1}.spm.spatial.preproc.data = {"
    lines += windows_anat_file
    lines += '};\n'
    lines += common_lines
    with open(m_script_file, 'w') as m_script_lines:
        m_script_lines.write(lines)
        

