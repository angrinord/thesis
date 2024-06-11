for ((counter=1; counter<=$1; counter++))
do
  echo "Starting " "$3" " Job" "$counter"/"$1"
  python pretrain_deepset.py -i "$counter" --batch_size "$2" --regime "$3"
  sleep 0.2
done

