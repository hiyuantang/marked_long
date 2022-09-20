python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=8_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 100 --epochs 140 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 8 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=10_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 100 --epochs 140 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 10 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=12_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 12 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=14_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 12 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=16_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 16 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=18_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 18 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=20_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 20 --optimizer adam
python train_model.py --checkpoint checkpoints/mazes_small_segment_adam_check/1 --output results/mazes_small_segment_adam --train_log recur_residual_network_width=2_depth=22_1.txt --model recur_residual_network_segment --dataset mazes_small --val_period 20 --lr 0.0005 --lr_factor 0.25 --lr_schedule 40 80 --epochs 120 --problem segment --save_json --save_period 50 --train_batch_size 50 --width 2 --depth 22 --optimizer adam