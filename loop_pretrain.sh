for ((counter=1; counter<=$1; counter++))
do
  echo "Queueing " "$2" " Job" "$counter"/"$1"
  sbatch rapidNAS_pretrain.sh "$counter" "$2"
  sleep 0.2
done

