# MINER
 Repo of paper "Interaction-level Membership Inference Attack against Recommender Systems with Long-tailed Distribution"

## Experiment steps

### 1. Target model training
- Run `python main_cke.py` to train the target CKE model. The model will be saved in `./KGAT_new/trained_model` directory. To train target model, should set `--train_shadow` as 0

### 2. Shadow model training
- Run `python main_cke.py` to train the shadow CKE model. The model will be saved in `./KGAT_new/trained_model` directory. To train shadow model, should set `--train_shadow` as 1

### 3. Membership inference attack
- Run `python attack_flow.py` to perform membership inference attack. Use `--gamma` to control $\gamma$ and use `--Ks` to control the number of recommended items.
