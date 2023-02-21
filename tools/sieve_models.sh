#! /bin/bash
cd /home/i53/student/jandl/repos/acronym/acronym/models/ &&
for i in *.obj;
do
sub=$(echo $i| cut -d'_' -f 2)
[ ! -f "../meshes/$sub.obj" ] &&
../../../Manifold/build/manifold ../models/$sub.obj ../temp/$sub.obj &&
../../../Manifold/build/simplify -i ../temp/$sub.obj -o ../meshes/$sub.obj -m -r 0.2;
done