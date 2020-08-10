m_size=(2048 4096 8192 16384)
d=rtx2070
f=${d}-result.csv

if [[ -f $f ]]
then
  printf "$f already exists.\nTerminate.\n"
  exit 0
else
  echo "Writing result to $f"
  echo "mnk,cuBLAS,PTX,GAS,GAS-mimic" >> $f
fi

for i in ${m_size[@]}
do
  ./hgemm $i >> $f
done