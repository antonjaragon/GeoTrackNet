# python geotracknet.py \
#   --mode=train \
#   --dataset_dir=data/florida_2021_processed \
#   --trainingset_name=train.pkl \
#   --latent_size=100 \
#   --batch_size=32 \
#   --num_samples=16 \
#   --learning_rate=0.0003


python geotracknet.py \
  --mode=save_logprob \
  --dataset_dir=./data/florida_2021_processed \
  --trainingset_name=train.pkl \
  --testset_name=valid.pkl \
  # --testset_name=test.pkl \
  --latent_size=100 \
  --batch_size=32 \
  --num_samples=16 \
  --learning_rate=0.0003


# python geotracknet.py \
#   --mode=local_logprob \
#   --dataset_dir=./data/florida_2021_processed \
#   --trainingset_name=train.pkl \
#   --testset_name=valid.pkl \
#   --latent_size=100 \
#   --batch_size=32 \
#   --num_samples=16 \
#   --learning_rate=0.0003 \


# python geotracknet.py \
#   --mode=contrario_detection \
#   --dataset_dir=./data \
#   --trainingset_name=train.pkl \
#   --testset_name=test.pkl \
#   --contrario_eps=1e-10 \
#   --latent_size=100 \
#   --batch_size=32 \
#   --num_samples=16 \
#   --learning_rate=0.0003
