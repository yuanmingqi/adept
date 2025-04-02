for ue in 1 2 3 4; do
    for seed in 1 2 3; do
        python  ppo_procgen_nue.py --update_epochs ${ue} --env_id miner     --seed ${seed} --device cuda:0 > logs/g0_${seed}.log 2>&1 &
        python  ppo_procgen_nue.py --update_epochs ${ue} --env_id ninja     --seed ${seed} --device cuda:0 > logs/g1_${seed}.log 2>&1 &
        wait
        python  ppo_procgen_nue.py --update_epochs ${ue} --env_id plunder   --seed ${seed} --device cuda:0 > logs/g0_${seed}.log 2>&1 &
        python  ppo_procgen_nue.py --update_epochs ${ue} --env_id starpilot --seed ${seed} --device cuda:0 > logs/g1_${seed}.log 2>&1 &
        wait
    done
done
