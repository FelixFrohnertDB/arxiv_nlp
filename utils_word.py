import logging
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
import re 
from tqdm import tqdm 
from gensim.models import KeyedVectors, Word2Vec
import gc 


def update_co_occurrences(word_year_list,word_co_occurrences):
    # Iterate through the words in the list
    word_list, year = word_year_list
    
    for word in word_list:
        # If the word is not already in the dictionary, add it with an empty list
        if word not in word_co_occurrences:
            word_co_occurrences[word] = {}
        
        # Add words from the list to the co-occurrence list for the current word
        for other_word in word_list:
            # if other_word != word and other_word not in word_co_occurrences[word]:
            #     word_co_occurrences[word].append(other_word)
            if other_word != word and other_word not in word_co_occurrences[word]:
                word_co_occurrences[word][other_word] = [year] 
            
            elif other_word != word and other_word in word_co_occurrences[word]:
                # word_co_occurrences[word][other_word][0] +=1
                word_co_occurrences[word][other_word].append(year)

def keep_words_with_underscore(input_string):
    # Define a regular expression pattern to match words with underscores
    pattern = r'\b\w*_[\w_]*\b'

    # Use re.findall to extract words that match the pattern
    matching_words = re.findall(pattern, input_string)

    # Join the matching words to form the final string
    result = ' '.join(matching_words)
    return result

def get_word_co_occurrences(filtered_concept_arr, ngram_abstracts, year_arr):
    # Step 1: Create the physical concept dictionary
    phys_concept_dict = {k: 1 for k in filtered_concept_arr}

    # Step 2: Process the abstracts to filter words
    ocurr_arr = []
    for abstract, year in zip(ngram_abstracts, year_arr):
        temp = keep_words_with_underscore(abstract)
        if temp.count(" ") > 0:
            temp = temp.split(" ")
            temp = [s for s in temp if s in phys_concept_dict]
            ocurr_arr.append([list(filter(("_").__ne__, temp)), year])

    # Step 3: Update word co-occurrences
    word_co_occurrences = {}
    for word_list in tqdm(ocurr_arr):
        update_co_occurrences(word_list, word_co_occurrences)

    return word_co_occurrences

def similarity_cosine(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two numpy arrays.

    Parameters:
    - arr1: np.ndarray
    - arr2: np.ndarray

    Returns:
    - float: Cosine similarity between arr1 and arr2.
    """
    # Check if both arrays are 1-Dimensional or 2-Dimensional.
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both input arrays must be 1-Dimensional.")

    # Compute the dot product of the arrays
    dot_product = np.dot(arr1, arr2)

    # Compute the magnitudes (norms) of the arrays
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)

    # Check for zero vectors to avoid division by zero
    if norm_arr1 == 0 or norm_arr2 == 0:
        raise ValueError("One or both of the vectors have zero magnitude.")

    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_arr1 * norm_arr2)
    
    return np.array([cosine_sim])

def get_co_occur_concept_pair_after_year_arr(word_co_occurrences: dict, first_occ_year: int, final_occ_year: int) -> np.ndarray:
    co_occur_concept_pair_arr = []
    for concept, v in word_co_occurrences.items():
        for co_concept, years in v.items():
            if np.min(years) >= first_occ_year and np.max(years) <= final_occ_year:
                co_occur_concept_pair_arr.append([concept,co_concept])
    return np.array(co_occur_concept_pair_arr)

def get_years_range(year_arr: np.ndarray, start: int, end: int) -> np.ndarray:
        return (np.unique(year_arr)[start:] if end == -0 
                                 else np.unique(year_arr)[start:end])

def get_baseline_1_embeddings(loaded_w2v: KeyedVectors, filtered_concept_arr: np.ndarray, embedding_dim: int = 128):
    """
    Extract embeddings for physical concepts from a Word2Vec model.

    Args:
        loaded_w2v (KeyedVectors): The preloaded Word2Vec model.
        phys_concept_dict (dict): Dictionary of physical concepts.
        embedding_dim (int): Dimensionality of the embeddings. Defaults to 128.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of concept embeddings.
            - np.ndarray: Array of concept identifiers.
    """
    c_dict = {}
    phys_concept_dict = {k: 1 for k in filtered_concept_arr}
    
    for c in phys_concept_dict:
        try:
            vec_enc = loaded_w2v.wv[c]  # Use direct indexing for simplicity
            c_dict[c] = vec_enc
        except KeyError:
            continue

    num_found = len(c_dict)
    num_missed = len(phys_concept_dict) - num_found
    print(f"Found {num_found} vectors, missed {num_missed} vectors.")
    
    if num_found == 0:
        return np.empty((0, embedding_dim)), np.empty((0,), dtype="<U55")

    c_encoding_arr = np.zeros((num_found, embedding_dim))
    c_inx_arr = np.zeros((num_found,), dtype="<U55")

    for cnt, (concept, encoding) in enumerate(c_dict.items()):
        c_encoding_arr[cnt] = encoding 
        c_inx_arr[cnt] = concept

    return c_encoding_arr, c_inx_arr

# def load_model_for_year(year):
#     return Word2Vec.load(f"saved_models/re_model_year_{year}.model")

# def get_method_embeddings(loaded_w2v: KeyedVectors, filtered_concept_arr: np.ndarray, embedding_dim: int = 128, load: bool = True):


#     if load:
#         c_inx_arr = np.memmap("saved_files/embedding_concept_arr.dat",shape=(10235,), dtype="<U55")
#         c_encoding_arr = np.memmap("saved_files/embedding_vector_arr.dat",shape=(10235, 30, 128), dtype=np.float64)

#     else:
#         # Initialize dictionaries and counters
#         c_dict = {}
#         phys_concept_dict = {k: 1 for k in filtered_concept_arr}
#         cnt_0, cnt_1 = 0, 0

#         # Get the unique years
#         unique_years = np.unique(year_arr)

#         # Iterate over each year and load the corresponding model
#         for year in tqdm(unique_years):
#             loaded_w2v = load_model_for_year(year)

#             # Iterate over each concept in the filtered concept dictionary
#             for c in phys_concept_dict:
#                 if c not in c_dict:
#                     c_dict[c] = {}

#                 # If the concept is already recorded for the current year, skip it
#                 if year in c_dict[c]:
#                     continue

#                 try:
#                     # Get the vector encoding for the concept
#                     vec_enc = loaded_w2v.wv.get_vector(c)
#                     c_dict[c][year] = vec_enc
#                     cnt_0 += 1
#                 except KeyError:  # Catch specific exception for missing key
#                     cnt_1 += 1
#                     pass

#         print(f"Found {cnt_0} vectors, missed {cnt_1} vectors.")

#         # Initialize lists for lengths and concept indices
#         len_arr, len_new_arr, concept_inx_arr = [], [], []

#         # Iterate over each concept to fill missing years with the first available vector
#         for c in tqdm(phys_concept_dict):
#             l = len(c_dict[c])
#             len_arr.append(l)

#             if l > 0:
#                 concept_inx_arr.append(c)
#                 success_years = sorted(c_dict[c].keys())
#                 first_success_year = success_years[0]

#                 # Backtrack and fill in the missing years with the first available vector
#                 for year in unique_years:
#                     if year < first_success_year:
#                         if year not in c_dict[c]:
#                             c_dict[c][year] = c_dict[c][first_success_year]
#                     else:
#                         break

#             len_new_arr.append(len(c_dict[c]))

#         concept_inx_arr = np.array(concept_inx_arr)

#         # Display the distribution of the number of years filled for each concept
#         print(np.unique(len_new_arr, return_counts=True))

#         # Prepare the encoding array
#         num_concepts = len(c_dict)
#         num_years = len(unique_years)
#         embedding_dim = 128

#         c_encoding_arr = np.zeros((num_concepts, num_years, embedding_dim))
#         c_inx_arr = []

#         # Fill the encoding array with vectors for each concept and year
#         for cnt, (concept, year_vectors) in enumerate(c_dict.items()):
#             c_encoding_arr[cnt] = np.array([year_vectors.get(year, np.zeros(embedding_dim)) for year in unique_years])
#             c_inx_arr.append(concept)

#         c_inx_arr = np.array(c_inx_arr)

        
#         filename = 'saved_files/embedding_concept_arr.dat'
#         memmap_array = np.memmap(filename, dtype=c_inx_arr.dtype, mode='w+', shape=c_inx_arr.shape)
#         print("shape:",c_inx_arr.shape)
#         memmap_array[:] = c_inx_arr
#         memmap_array.flush()

#         filename = 'saved_files/embedding_vector_arr.dat'
#         memmap_array = np.memmap(filename, dtype=c_encoding_arr.dtype, mode='w+', shape=c_encoding_arr.shape)
#         print("shape:",c_encoding_arr.shape)
#         memmap_array[:] = c_encoding_arr
#         memmap_array.flush()

def load_model_for_year(year):
    """Load Word2Vec model for a specific year."""
    return Word2Vec.load(f"saved_models/re_model_year_{year}.model")

def get_method_embeddings(filtered_concept_arr: np.ndarray, year_arr: np.ndarray, embedding_dim: int = 128, load: bool = True):
    """Get method embeddings, either by loading from files or processing the data."""
    
    if load:
        # Load pre-saved embedding arrays
        c_inx_arr = np.memmap("saved_files/embedding_concept_arr.dat", shape=(10235,), dtype="<U55")
        c_encoding_arr = np.memmap("saved_files/embedding_vector_arr.dat", shape=(10235, 30, 128), dtype=np.float64)
    else:
        # Initialize dictionaries and counters
        c_dict = {}
        phys_concept_dict = {k: 1 for k in filtered_concept_arr}
        cnt_0, cnt_1 = 0, 0

        # Get unique years
        unique_years = np.unique(year_arr)

        # Load models and get vectors
        for year in tqdm(unique_years, desc="Loading models and vectors"):
            loaded_w2v = load_model_for_year(year)
            for c in phys_concept_dict:
                if c not in c_dict:
                    c_dict[c] = {}
                if year in c_dict[c]:
                    continue
                try:
                    vec_enc = loaded_w2v.wv[c]
                    c_dict[c][year] = vec_enc
                    cnt_0 += 1
                except KeyError:
                    cnt_1 += 1

        print(f"Found {cnt_0} vectors, missed {cnt_1} vectors.")

        # Fill missing years
        concept_inx_arr = []
        for c in tqdm(phys_concept_dict, desc="Filling missing years"):
            if c_dict[c]:
                concept_inx_arr.append(c)
                success_years = sorted(c_dict[c].keys())
                first_success_year = success_years[0]
                for year in unique_years:
                    if year < first_success_year:
                        c_dict[c][year] = c_dict[c][first_success_year]
                    else:
                        break

        concept_inx_arr = np.array(concept_inx_arr)

        # Prepare the encoding array
        num_concepts = len(concept_inx_arr)
        num_years = len(unique_years)

        c_encoding_arr = np.zeros((num_concepts, num_years, embedding_dim))
        c_inx_arr = []

        for cnt, concept in enumerate(tqdm(concept_inx_arr, desc="Preparing encoding array")):
            year_vectors = c_dict[concept]
            c_encoding_arr[cnt] = np.array([year_vectors.get(year, np.zeros(embedding_dim)) for year in unique_years])
            c_inx_arr.append(concept)

        c_inx_arr = np.array(c_inx_arr)

        # Save the arrays using memmap
        save_memmap("saved_files/embedding_concept_arr.dat", c_inx_arr)
        save_memmap("saved_files/embedding_vector_arr.dat", c_encoding_arr)

        # Clear the garbage collector
        gc.collect()

        # Load the memmap arrays
        c_inx_arr = np.memmap("saved_files/embedding_concept_arr.dat", dtype="<U55", mode='r', shape=c_inx_arr.shape)
        c_encoding_arr = np.memmap("saved_files/embedding_vector_arr.dat", dtype=np.float64, mode='r', shape=c_encoding_arr.shape)

    return c_inx_arr, c_encoding_arr

def save_memmap(filename, array):
    """Save an array to a memmap file."""
    memmap_array = np.memmap(filename, dtype=array.dtype, mode='w+', shape=array.shape)
    memmap_array[:] = array[:]
    memmap_array.flush()
    print(f"Saved memmap file '{filename}' with shape {array.shape}")


def compute_roc(model, dataloader, filename=""):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels,_ ,_ in dataloader:
            outputs = model(data.float(), )
            probs = outputs.cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    if filename != "":
        np.save(f"saved_files/fpr_{filename}.npy", fpr)
        np.save(f"saved_files/tpr_{filename}.npy", tpr)
        print("saved")
        return 0 
        
    return fpr,tpr, roc_auc

def plot_roc(model, dataloader):
    
    fpr,tpr, roc_auc = compute_roc(model, dataloader)
    # Plot ROC curve
    plt.figure(figsize=(3.5, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def compute_acc(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            outputs = model(data.float())
            probs = outputs.cpu().numpy()
            predicted = (outputs > 0.5).float()  # Use 0.5 as threshold for binary classification
            
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total

    return accuracy

    # def replace_strings_with_indices(data, concept_to_index):
#     # Create a new dictionary to store the converted data
#     indexed_data = {}
    
#     # Iterate through the outer dictionary
#     for concept1, nested_dict in data.items():
#         # Replace the concept1 string with its index
#         # print(concept_to_index[concept1])
#         index1 = concept_to_index[concept1][0]
#         indexed_data[index1] = {}
        
#         # Iterate through the inner dictionary
#         for concept2, years in nested_dict.items():
#             # Replace the concept2 string with its index
            
#             index2 = concept_to_index[concept2][0]
#             indexed_data[index1][index2] = years
    
#     return indexed_data


# concept_to_indices = {concept: np.where(concept_filtered_arr == concept)[0] for concept in np.unique(concept_filtered_arr)}
# index_co_occurrences = replace_strings_with_indices(word_co_occurrences, concept_to_indices)