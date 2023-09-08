for counter in $1
do
  echo "Queueing Job" "$counter"/"$1"
  sbatch evaluate.sh "$counter"
  sleep 0.2
done
