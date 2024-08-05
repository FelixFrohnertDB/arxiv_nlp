# arxiv_nlp

Files:
- arxiv-metadata-oai-snapshot.json
- arxiv_preprocessed.csv id abstract date

- arxiv_atomic_concept.txt
- arxiv_optics_concept.txt
- arxiv_quantum_concept.txt 

- ngram_abstracts.npy
- overlapping_filtered_5_concepts.npy
- year_arr.npy

- model_year_X.model 

- embedding_concept_arr.dat
- embedding_vector_arr.dat


    train_pos_inx_pair_arr_5_3_3 = np.load(f"saved_files/train_pos_inx_pair_arr_{seq_length}_{out_length}_3.npy")
    train_neg_inx_pair_arr_5_3_3 = np.load(f"saved_files/train_neg_inx_pair_arr_{seq_length}_{out_length}_3.npy")