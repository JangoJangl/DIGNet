#! /bin/bash

cd /home/i53/student/jandl/repos/acronym/acronym/grasps/ &&
for i in *.h5; 
do
folder=${i%%_*}
sub=$(echo $i| cut -d'_' -f 2)
[ ! -d "../meshes/$folder" ] && mkdir -p "../meshes/$folder"
mv ../meshes/$sub.obj ../meshes/$folder/$sub.obj

done

