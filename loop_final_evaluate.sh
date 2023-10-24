for ((counter=1; counter<=$1; counter++))
do
  echo "Queueing Surrogate Job" "$counter"/"$1"
  sbatch final_evaluate.sh "$counter"
  sleep 0.2
done

