for ((counter=1; counter<=$1; counter++))
do
  echo "Queueing " "$3" " Job" "$counter"/"$1"
  sbatch deepset_pretrain.sh "$counter" "$2" "$3"
  sleep 0.2
done

