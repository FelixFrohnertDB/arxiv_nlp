import networkx as nx
from datetime import datetime, date
import pickle
import numpy as np 
from scipy import sparse
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size):
        """
        Fully Connected layers
        """
        super(MLP, self).__init__()

        self.semnet = nn.Sequential( # very small network for tests
            nn.Linear(input_size, 128), nn.PReLU(), 
            nn.Dropout(0.25), nn.Linear(128, 64), nn.PReLU(), 
            nn.Dropout(0.25), nn.Linear( 64, 64), nn.PReLU(), 
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )


    def forward(self, x):
        """
        Pass throught network
        """
        res = self.semnet(x)

        return res

def create_graph_from_dict(data):
    # Create an empty graph
    G = nx.Graph()
    
    # Iterate through the outer dictionary
    for concept1, nested_dict in data.items():
        # Iterate through the inner dictionary
        for concept2, years in nested_dict.items():
            # If the edge already exists, add to the weight and append years
            if G.has_edge(concept1, concept2):
                G[concept1][concept2]['weight'] += len(years)
                G[concept1][concept2]['years'].extend(years)
            else:
                # Add a new edge with the weight and list of years
                G.add_edge(concept1, concept2, weight=len(years), years=years)
    
    # Remove duplicate years
    for u, v in G.edges():
        G[u][v]['years'] = list(set(G[u][v]['years']))
    
    return G

def convert_years_to_days_since_1990(years):
    base_date = datetime(1990, 1, 1)
    days_since_1990 = [(datetime(year, 1, 1) - base_date).days for year in years]
    return days_since_1990

def save_graph_edges(graph, filename):
    # Create a list of concepts and assign indices
    concepts = list(graph.nodes())
    concept_indices = {concept: idx for idx, concept in enumerate(concepts)}
    
    # Create the edge list in the specified format
    edge_list = []
    for u, v, data in graph.edges(data=True):
        if u>v:
            pass#print("no...", u,v,data)
        idx1 = concept_indices[u]
        idx2 = concept_indices[v]
        years = data['years']
        days = convert_years_to_days_since_1990(years)
        for day in days:
            edge_list.append([idx1, idx2, day])
    
    # Save the edge list as a .pkl file
    with open(filename, 'wb') as f:
        pickle.dump(edge_list, f)

def create_training_data(full_graph,year_start,years_delta,NUM_OF_VERTICES,min_edges=1,edges_used=500000,vertex_degree_cutoff=10):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param min_edges: minimal number of edges that is considered
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """
    print('\n')
    print('in create_training_data ')
    print('Creating the following data: ')
    
    print('    year_start: ', year_start)
    print('    years_delta: ', years_delta)
    print('    min_edges: ', min_edges)
    print('    edges_used: ', edges_used)
    print('    vertex_degree_cutoff: ', vertex_degree_cutoff)

    years=[year_start, year_start+years_delta]    
    day_origin = date(1990,1,1)

    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
    all_degs=[]
    for yy in years:
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        G_curr=nx.from_scipy_sparse_array(adj_mat_sparse_curr, parallel_edges=False, create_using=nx.MultiGraph)

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)

        print('          num of edges: ', G_curr.number_of_edges())
        print('    Done: Create Graph for ', yy)
        

    all_degs=np.array(all_G[0].degree)[:,1]

    ## Create all edges to be predicted
    all_vertices=np.array(range(NUM_OF_VERTICES))
    
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.
    print("\n")
    print("all_vertices: ", len(all_vertices))
    print('len(vertex_large_degs): ',len(vertex_large_degs))

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]

    time_start=time.time()
    added_pairs = set()
    while len(unconnected_vertex_pairs)<edges_used:        
        i1,i2=random.sample(range(len(vertex_large_degs)), 2)
        
        v1=vertex_large_degs[i1]
        v2=vertex_large_degs[i2]

        if v1!=v2 and not all_G[0].has_edge(v1,v2):

            pair = (v1, v2) # currently keep the case (v2, v1)
            if pair not in added_pairs:

                if len(unconnected_vertex_pairs)%10**6==0:
                    time_end=time.time()
                    print('    edge progress (',time_end-time_start,'sec): ',len(unconnected_vertex_pairs)/10**6,'M/',edges_used/10**6,'M')
                    time_start=time.time()

                is_bigger=False
                if all_G[1].has_edge(v1,v2):
                    curr_weight=all_G[1].get_edge_data(v1,v2)[0]['weight']
                    if curr_weight>=min_edges:
                        is_bigger=True

                unconnected_vertex_pairs.append((v1,v2))
                unconnected_vertex_pairs_solution.append(is_bigger)
                added_pairs.add(pair)

    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))

    all_edge_list=np.array(all_edge_lists[0]) ## the graph of year_start
    
    print('unconnected_vertex_pairs_solution: ',sum(unconnected_vertex_pairs_solution))
    
    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution


def create_training_data_biased(full_graph,year_start,years_delta,NUM_OF_VERTICES,min_edges=1,edges_used=500000,vertex_degree_cutoff=10,data_source=''):
    """
    :param full_graph: Full graph, numpy array dim(n,3) [vertex 1, vertex 2, time stamp]
    :param year_start: year of graph
    :param years_delta: distance for prediction in years (prediction on graph of year_start+years_delta)
    :param min_edges: minimal number of edges that is considered
    :param edges_used: optional filter to create a random subset of edges for rapid prototyping (default: 500,000)
    :param vertex_degree_cutoff: optional filter, for vertices in training set having a minimal degree of at least vertex_degree_cutoff  (default: 10)
    :return:

    all_edge_list: graph of year_start, numpy array dim(n,2)
    unconnected_vertex_pairs: potential edges for year_start+years_delta
    unconnected_vertex_pairs_solution: numpy array with integers (0=unconnected, 1=connected), solution, length = len(unconnected_vertex_pairs)
    """
    with open(data_source+"_logs.txt", "a") as myfile:
        myfile.write('\nin create_training_data_biased')      
    print('\n')
    print('in create_training_data_biased ')
    print('Creating the following data: ')
    
    print('    year_start: ', year_start)
    print('    years_delta: ', years_delta)
    print('    min_edges: ', min_edges)
    print('    edges_used: ', edges_used)
    print('    vertex_degree_cutoff: ', vertex_degree_cutoff)

    years=[year_start,year_start+years_delta]    
    day_origin = date(1990, 1, 1)

    all_G=[]
    all_edge_lists=[]
    all_sparse=[]
    all_degs=[]
    for yy in years:
        with open(data_source+"_logs.txt", "a") as myfile:
            myfile.write('\n    Create Graph for '+str(yy))    
        print('    Create Graph for ', yy)
        day_curr=date(yy,12,31)
        all_edges_curr=full_graph[full_graph[:,2]<(day_curr-day_origin).days]
        adj_mat_sparse_curr = sparse.csr_matrix((np.ones(len(all_edges_curr)), (all_edges_curr[:,0], all_edges_curr[:,1])), shape=(NUM_OF_VERTICES,NUM_OF_VERTICES))
        G_curr=nx.from_scipy_sparse_array(adj_mat_sparse_curr, parallel_edges=False, create_using=nx.MultiGraph)

        all_G.append(G_curr)
        all_sparse.append(adj_mat_sparse_curr)
        all_edge_lists.append(all_edges_curr)

        print('          num of edges: ', G_curr.number_of_edges())
        print('    Done: Create Graph for ', yy)
        

    all_degs=np.array(all_G[0].degree)[:,1]

    ## Create all edges to be predicted
    all_vertices=np.array(range(NUM_OF_VERTICES))
    vertex_large_degs=all_vertices[all_degs>=vertex_degree_cutoff] # use only vertices with degrees larger than 10.
    print(f"\nlen(all_vertices): {len(all_vertices)}")
    print(f"len(vertex_large_degs): {len(vertex_large_degs)}\n")
    with open(data_source+"_logs.txt", "a") as myfile:
        myfile.write('\nlen(vertex_large_degs): '+str(len(vertex_large_degs)))

    unconnected_vertex_pairs=[]
    unconnected_vertex_pairs_solution=[]

    time_start=time.time()
    cT=0
    cF=0
    old_c=0
    added_pairs = set()
    ## equal number of false and true samples
    while (cT<(edges_used/2)) or (cF<(edges_used/2)):        
        i1,i2=random.sample(range(len(vertex_large_degs)), 2)
        
        v1=vertex_large_degs[i1]
        v2=vertex_large_degs[i2]

        if v1!=v2 and not all_G[0].has_edge(v1,v2): ## unconnected in 2018
            
            pair = (v1, v2) # currently keep the case (v2, v1)
            if pair not in added_pairs:
            
                if len(unconnected_vertex_pairs)%10**4==0 and len(unconnected_vertex_pairs)!=old_c:
                    time_end=time.time()
                    
                    print(f"    edge progress ({time_end-time_start}sec): {len(unconnected_vertex_pairs)/10**6}M/{edges_used/10**6}M; True: {cT}; False: {cF}")
                    
                    with open(data_source+"_logs.txt", "a") as myfile:
                        myfile.write('\n    edge progress ('+str(time_end-time_start)+'sec): '+str(len(unconnected_vertex_pairs)/10**6)+'M/'+str(edges_used/10**6)+'M '+str(cT)+' '+str(cF))
                    old_c=len(unconnected_vertex_pairs)
                    time_start=time.time()
                

                is_bigger=False
                if all_G[1].has_edge(v1,v2): ## connected in 2021
                    curr_weight=all_G[1].get_edge_data(v1,v2)[0]['weight']
                    if curr_weight>=min_edges:
                        is_bigger=True

                if is_bigger==False and cF<edges_used/2:
                    unconnected_vertex_pairs.append((v1,v2))
                    unconnected_vertex_pairs_solution.append(is_bigger)
                    cF+=1
                if is_bigger==True and cT<edges_used/2:
                    unconnected_vertex_pairs.append((v1,v2))
                    unconnected_vertex_pairs_solution.append(is_bigger)
                    cT+=1
                    #print("yes...")
                added_pairs.add(pair)
            
            
    print("(edges_used/2), cT, cF: ", (edges_used/2), cT, cF)
    with open(data_source+"_logs.txt", "a") as myfile:
        myfile.write('\nnearly done here')

    unconnected_vertex_pairs=np.array(unconnected_vertex_pairs)
    unconnected_vertex_pairs_solution=np.array(list(map(int, unconnected_vertex_pairs_solution)))

    all_edge_list=np.array(all_edge_lists[0]) # graph of year_start
    
    print('unconnected_vertex_pairs_solution: ',sum(unconnected_vertex_pairs_solution))
    
    return all_edge_list, unconnected_vertex_pairs, unconnected_vertex_pairs_solution



def calculate_ROC(data_vertex_pairs,data_solution, train_info="", save=False, directory=""):
    data_solution=np.array(data_solution)
    data_vertex_pairs_sorted=data_solution[data_vertex_pairs]
    
    xpos=[0]
    ypos=[0]
    ROC_vals=[]
    for ii in range(len(data_vertex_pairs_sorted)):
        if data_vertex_pairs_sorted[ii]==1:
            xpos.append(xpos[-1])
            ypos.append(ypos[-1]+1)
        if data_vertex_pairs_sorted[ii]==0:
            xpos.append(xpos[-1]+1)
            ypos.append(ypos[-1])      
            ROC_vals.append(ypos[-1])
    
        # # # # # # # # # # # # # # # 
        # 
        # We normalize the ROC curve such that it starts at (0,0) and ends at (1,1).
        # Then our final metric of interest is the Area under that curve.
        # AUC is between [0,1].
        # AUC = 0.5 is acchieved by random predictions
        # AUC = 1.0 stands for perfect prediction.
    
    ROC_vals=np.array(ROC_vals)/max(ypos)
    ypos=np.array(ypos)/max(ypos)
    xpos=np.array(xpos)/max(xpos)
    AUC=sum(ROC_vals)/len(ROC_vals)
    if save:
        np.save(f"saved_files/fpr_{directory}.npy",xpos)
        np.save(f"saved_files/tpr_{directory}.npy",ypos)

    plt.title(f"AUC: {AUC}; {train_info}")
    plt.plot(xpos, ypos)
    plt.show()
    plt.close()

    return AUC

def calculate_accuracy(data_vertex_pairs, data_solution, train_info="", save=False):
    data_solution = np.array(data_solution)
    data_vertex_pairs_sorted = data_solution[data_vertex_pairs]
    
    correct_predictions = 0
    total_predictions = len(data_vertex_pairs_sorted)
    
    for ii in range(total_predictions):
        # Assuming `data_vertex_pairs_sorted` holds binary predictions (0 or 1)
        if data_vertex_pairs_sorted[ii] == data_solution[ii]:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    if save:
        np.save(f"saved_files/accuracy_{train_info}.npy", accuracy)
    
    plt.title(f"Accuracy: {accuracy:.4f}; {train_info}")
    plt.bar(["Correct Predictions", "Total Predictions"], [correct_predictions, total_predictions])
    plt.ylabel("Count")
    plt.show()
    plt.close()

    return accuracy


def train_model(model_semnet, data_train0, data_train1, data_test0, data_test1, lr_enc, batch_size, data_source, year_start, year_delta):
    """
    Training the neural network
    """            
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size_of_loss_check=2000
    
    optimizer_predictor = torch.optim.Adam(model_semnet.parameters(), lr=lr_enc)
    
    data_train0=torch.tensor(data_train0, dtype=torch.float).to(device)
    data_test0=torch.tensor(data_test0, dtype=torch.float).to(device)
    
    data_train1=torch.tensor(data_train1, dtype=torch.float).to(device)
    data_test1=torch.tensor(data_test1, dtype=torch.float).to(device)

    test_loss_total=[]
    moving_avg=[]
    criterion = torch.nn.MSELoss()
    
    # There are much more vertex pairs that wont be connected (0) rather than ones
    # that will be connected (1). However, we observed that training with an equally weighted
    # training set (same number of examples for (0) and (1)) results in more stable training.
    # (Imaging we have 1.000.000 nonconnected and 10.000 connected)
    #
    # For that reason, we dont have true 'episodes' (where each example from the training set
    # has been used in the training). Rather, in each of our iteration, we sample batch_size
    # random training examples from data_train0 and from data_train1.
    
    for iteration in range(50000): # should be much larger, with good early stopping criteria
        model_semnet.train()
        data_sets=[data_train0,data_train1]
        total_loss=0
        for idx_dataset in range(len(data_sets)):
            idx = torch.randint(0, len(data_sets[idx_dataset]), (batch_size,))
            data_train_samples = data_sets[idx_dataset][idx]
            calc_properties = model_semnet(data_train_samples)
            curr_pred=torch.tensor([idx_dataset] * batch_size, dtype=torch.float).to(device)
            real_loss = criterion(calc_properties, curr_pred)
            total_loss += torch.clamp(real_loss, min = 0., max = 50000.).double()

        optimizer_predictor.zero_grad()
        total_loss.backward()
        optimizer_predictor.step()

        # Evaluating the current quality.
        with torch.no_grad():
            model_semnet.eval()
            # calculate train set
            eval_datasets=[data_train0,data_train1,data_test0,data_test1]
            all_real_loss=[]
            for idx_dataset in range(len(eval_datasets)):
                eval_datasets[idx_dataset]
                calc_properties = model_semnet(eval_datasets[idx_dataset][0:size_of_loss_check])        
                curr_pred=torch.tensor([idx_dataset%2] * len(eval_datasets[idx_dataset][0:size_of_loss_check]), dtype=torch.float).to(device)
                real_loss = criterion(calc_properties, curr_pred)
                all_real_loss.append(real_loss.detach().cpu().numpy())
             
            test_loss_total.append(np.mean(all_real_loss[2])+np.mean(all_real_loss[3]))

            if iteration%50==0:
                info_str='iteration: '+str(iteration)+' - train loss: '+str(np.mean(all_real_loss[0])+np.mean(all_real_loss[1]))+'; test loss: '+str(np.mean(all_real_loss[2])+np.mean(all_real_loss[3]))
                print('    train_model: ',info_str)
                with open(data_source+"_logs.txt", "a") as myfile:
                    myfile.write('\n    train_model: '+info_str)

            if test_loss_total[-1]==min(test_loss_total): ## store the best model
                model_semnet.eval()
                torch.save(model_semnet, f'saved_ml_models/baseline_3_full_{str(year_start)}_{str(year_delta)}')
                torch.save(model_semnet.state_dict(), f'saved_ml_models/baseline_3_state_{str(year_start)}_{str(year_delta)}.pt')  
                model_semnet.train()
                with open(data_source+"_logs.txt", "a") as myfile:
                    myfile.write('\n    stored model ')

            if len(test_loss_total)>200: # early stopping
                test_loss_moving_avg=sum(test_loss_total[-100:])
                moving_avg.append(test_loss_moving_avg)
                if len(moving_avg)>10:
                    
                    if moving_avg[-1]>moving_avg[-5] and moving_avg[-1]>moving_avg[-25]:
                        print('    Early stopping kicked in')
                        break

    plt.plot(test_loss_total)
    plt.title(f"test loss: year_start: {str(year_start)}, year_delta: {str(year_delta)}")
    plt.show()
    
    plt.plot(test_loss_total[500:])
    plt.title(f"test loss, last 500: year_start: {str(year_start)}, year_delta: {str(year_delta)}")
    plt.show()    
    
    plt.plot(moving_avg)
    plt.title(f'moving avg loss: year_start: {str(year_start)}, year_delta: {str(year_delta)}')
    plt.show()
    plt.close()

    return True


def compute_all_properties(all_sparse, AA02,AA12,AA22,all_degs0,all_degs1,all_degs2,all_degs02,all_degs12,all_degs22,v1,v2):
    """
    Computes hand-crafted properties for one vertex in vlist
    """
    all_properties=[]

    all_properties.append(all_degs0[v1]) # 0
    all_properties.append(all_degs0[v2]) # 1
    all_properties.append(all_degs1[v1]) # 2
    all_properties.append(all_degs1[v2]) # 3
    all_properties.append(all_degs2[v1]) # 4
    all_properties.append(all_degs2[v2]) # 5
    all_properties.append(all_degs02[v1]) # 6
    all_properties.append(all_degs02[v2]) # 7
    all_properties.append(all_degs12[v1]) # 8
    all_properties.append(all_degs12[v2]) # 9
    all_properties.append(all_degs22[v1]) # 10
    all_properties.append(all_degs22[v2]) # 11

    all_properties.append(AA02[v1,v2]) # 12
    all_properties.append(AA12[v1,v2]) # 13
    all_properties.append(AA22[v1,v2]) # 14    

    return all_properties



def compute_all_properties_of_list(all_sparse,vlist,data_source):
    """
    Computes hand-crafted properties for all vertices in vlist
    """
    time_start=time.time()
    AA02=all_sparse[0]**2
    AA02=AA02/AA02.max()
    AA12=all_sparse[1]**2
    AA12=AA12/AA12.max()
    AA22=all_sparse[2]**2
    AA22=AA22/AA22.max()
    
    all_degs0=np.array(all_sparse[0].sum(0))[0]
    if np.max(all_degs0)>0:
        all_degs0=all_degs0/np.max(all_degs0)
        
    all_degs1=np.array(all_sparse[1].sum(0))[0]
    if np.max(all_degs1)>0:
        all_degs1=all_degs1/np.max(all_degs1)
    
    all_degs2=np.array(all_sparse[2].sum(0))[0]
    if np.max(all_degs2)>0:
        all_degs2=all_degs2/np.max(all_degs2)

    all_degs02=np.array(AA02[0].sum(0))[0]
    if np.max(all_degs02)>0:
        all_degs02=all_degs02/np.max(all_degs02)
        
    all_degs12=np.array(AA12[1].sum(0))[0]
    if np.max(all_degs12)>0:
        all_degs12=all_degs12/np.max(all_degs12)
        
    all_degs22=np.array(AA22[2].sum(0))[0]
    if np.max(all_degs22)>0:
        all_degs22=all_degs22/np.max(all_degs22)
    
    all_properties=[]
    print('    Computed all matrix squares, ready to ruuuumbleeee...')
    for ii in range(len(vlist)):
        vals=compute_all_properties(all_sparse,
                                    AA02,
                                    AA12,
                                    AA22,
                                    all_degs0,
                                    all_degs1,
                                    all_degs2,
                                    all_degs02,
                                    all_degs12,
                                    all_degs22,
                                    vlist[ii][0],
                                    vlist[ii][1])

        all_properties.append(vals)
        if ii%10**4==0:
            print('    compute_all_properties_of_list progress: (',time.time()-time_start,'sec) ',ii/10**6,'M/',len(vlist)/10**6,'M')

            with open(data_source+"_logs.txt", "a") as myfile:
                myfile.write('\n    compute_all_properties_of_list progress: ('+str(time.time()-time_start)+'sec) '+str(ii/10**6)+'M/'+str(len(vlist)/10**6)+'M')

            time_start=time.time()

    return all_properties


def flatten(t):
    return [item for sublist in t for item in sublist]