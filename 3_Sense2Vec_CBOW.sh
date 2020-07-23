python '3. Sense2Vec - CBOW.py' \
  --lr=3e-3 \
  --bs=2500 \
  --seq_len=5 \
  --epochs=8 \
  --device=cuda:4 \
  --input_corpus=data/postprocessed/ds_blogs.txt \
  --embeddings_size=150 \
  --target_vectors=200 \
  --mlflow_host=http://192.168.113.181:5000 \
  --mlflow_experiment=Sense2Vec_blogs_test \
  --model_pickles_dir_path=data/models \
  --dataset_pickle_path=data/datasets \
  --minimal_token_occurences=5
