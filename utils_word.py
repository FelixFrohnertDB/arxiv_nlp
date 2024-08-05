import logging
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
import re 
from tqdm import tqdm 

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