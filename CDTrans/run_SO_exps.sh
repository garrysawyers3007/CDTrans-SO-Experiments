# bash scripts/pretrain/officehome/run_officehome_Ar_1.sh deit_base>logs/Ar_mixup_0.1_1.txt
# bash scripts/pretrain/officehome/run_officehome_Ar_2.sh deit_base>logs/Ar_mixup_0.1_2.txt
# bash scripts/pretrain/officehome/run_officehome_Ar_3.sh deit_base>logs/Ar_mixup_0.1_3.txt
# bash scripts/pretrain/officehome/run_officehome_Pr.sh deit_base>logs/train_P2R.txt
# bash scripts/pretrain/officehome/run_officehome_Rw.sh deit_base>logs/train_R2A.txt
bash scripts/pretrain/officehome/run_officehome_Ar.sh deit_base
bash scripts/pretrain/officehome/run_officehome_Cl.sh deit_base
bash scripts/pretrain/officehome/run_officehome_Pr.sh deit_base
bash scripts/pretrain/officehome/run_officehome_Rw.sh deit_base
