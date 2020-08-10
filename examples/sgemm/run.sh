m_size=(2048 4096 8192 10240)
d=v100
f=${d}-result.csv

if [[ -f $f ]]
then
  printf "$f already exists.\nTerminate.\n"
  exit 0
else
  echo "Writing result to $f"
  echo "mnk,cuda,cuBLAS,PTX,GAS,GAS-mimic,GAS-yield" >> $f
fi

for i in ${m_size[@]}
do
  ./sgemm $i >> $f
done
