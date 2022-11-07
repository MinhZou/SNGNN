for model in SNGNN_Plus
do
  for dataset in PubMed CiteSeer Chameleon Texas Cora Wisconsin Cornell PubMed Squirrel Actor
  do
    for lr in 0.1
    do
      for weight_decay in 0.0005
      do
        for dropout in 0
        do
          for is_remove_self_loops in 0
          do
            for hidden_channels in 64
            do
              for num_layers in 1
              do
                for top_k in 1
                do
                  for thr in 0.99
                  do
                    for patience in 300
                    do
                      for part_id in 0 1 2 3 4 5 6 7 8 9
                      do
                        CUDA_VISIBLE_DEVICES=1 python train.py --config ./config/config-test.yaml \
                          --work-dir ./work_dir-$dataset-$model \
                          --seed 1234 \
                          --epochs 2000 \
                          --patience $patience \
                          --model $model \
                          --dataset $dataset \
                          --lr $lr \
                          --weight_decay $weight_decay \
                          --dropout $dropout \
                          --hidden_channels $hidden_channels \
                          --num_layers $num_layers \
                          --part_id $part_id \
                          --data_splits \
                          --top_k $top_k\
                          --thr $thr \
                          --is_remove_self_loops $is_remove_self_loops
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done