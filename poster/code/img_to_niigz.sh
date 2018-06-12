#-
echo .img -> .nii.gz in
pattern=$HOME/nilearn_data/mrm_2010/*.img
echo pattern
for fichier in $pattern; do
    #echo Traitement de $fichier
    echo $fichier
    fslchfiletype NIFTI_GZ $fichier
done
#-
