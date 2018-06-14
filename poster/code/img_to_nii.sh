#-
echo .img -> .nii in
pattern=/home/Pmamobipet/0_Dossiers-Personnes/Salma/mrm_2010/reoriented_template/*.img
echo pattern
for fichier in $pattern; do
    #echo Traitement de $fichier
    echo $fichier
    fslchfiletype NIFTI $fichier
done
#-
