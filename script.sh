arr_n_agents=(10 15 20)
arr_distr=(poisson uniform)

for i in "${arr_n_agents[@]}"
do
    for j in "${arr_distr[@]}"
    do
        k=5
        while (( k < i ))
        do
            python main.py --n_steps=500 --n_agents="$i" --n_rcv="$k" --distr="$j" > /dev/null 2>&1 &
            tail --pid=$! -f /dev/null
            echo $k
            k=$(( k - 1 ))
        done
    done
done