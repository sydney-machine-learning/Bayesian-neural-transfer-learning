cd scripts/bntl
echo "moving to bntl directoty"

for ratio in 0.1 0.25 0.50 0.75 1.0 
	do
	echo "running experiment for ratio $ratio.."
	python3 ldbntl.py --langevin --problem 3 --langevin-ratio $ratio --run-id 1
done


