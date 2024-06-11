for ((counter=1; counter<=$1; counter++))
do
  echo "Starting " "$2" " Job" "$counter"/"$1"
  python cluster_evaluate.py -i "$counter" --regime "$2"
  sleep 0.2
done

