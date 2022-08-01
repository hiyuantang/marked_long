#!/bin/bash
n_machines=50
num_images=200
#imgs_path='/gpfs/data/tserre/data/pathfinder/'
imgs_path='/home/AD/yutang/pathfinder21_dataset/'
script_name='snakes2_wrapper.py'
username='yutang'

# submit job
PARTITION='batch' # batch # bibs-smp # bibs-gpu # gpu # small-batch
QOS='bibs-tserre-condo' # pri-jk9

for i_machine in $(seq 1 $n_machines); do
sbatch -J "PATHFINDER-$script_name[$i_machine]" <<EOF
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -n 2
#SBATCH -t 4:00:00
#SBATCH --mem=8G
#SBATCH --begin=now
#SBATCH --account=$QOS
#SBATCH --output=/home/AD/yutang/scratch/$username/slurm/pathfinder-$i_machine.out
#SBATCH --error=/home/AD/yutang/scratch/$username/slurm/pathfinder-$i_machine.out

echo "Starting job $i_machine on $HOSTNAME"
LC_ALL=en_US.utf8 \
module load boost ffmpeg/1.2

python $script_name $n_machines $i_machine $num_images $imgs_path
EOF
done

