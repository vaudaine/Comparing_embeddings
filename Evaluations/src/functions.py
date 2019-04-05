import numpy as np
import os
import networkx as nx
import time
import collections
import random
import scipy.sparse as sparse
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.manifold import MDS
from pandas import DataFrame
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, normalize
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr




def embedding_distance(embedding, metric='euclidean'):
    """Returns a 2D-array whose [i][j] is the distance between vector i and j of embedding """
    return cdist(embedding, embedding, metric)

def neighbors_dissimilarity(adjacency_matrix, embedding, metric='euclidean'):
    """For every node i in the graph G, computes neighbors_list the list of neighbors of i in G,
    then computes the list indices of the degree(i) nearest nodes in the embedding. 
    Finally, computes the fraction of nodes that are not in common in the lists.
        
    For example, node '5' has three neighbors in G ['0','2','3']. 
    In the embedding, the three nearest nodes to '5' are ['0','1','3']
    Fraction of nodes not in common = 1 - fraction of nodes in common = 1 - 2/3 = 1-3
    This is 1-value as compared with our paper."""

    dist = embedding_distance(embedding, metric)
    D = []
    N = len(adjacency_matrix)
    if N > 2000:
        A = [i for i in range(N)]
        fraction_of_sampled_nodes = 0.1
        number_of_sampled_nodes = int(N*fraction_of_sampled_nodes)
        list_of_random_nodes = random.sample(A, number_of_sampled_nodes)
    else:
        list_of_random_nodes = range(N)

    
    for j in list_of_random_nodes: #for every node
        maximum = np.copy(dist[j]) #computes the distances between j and every node in the graph
        maximum.sort() #order those distances
        rank = int(sum(adjacency_matrix[j])) #number of neighbors (degree of node j)
        maxi = maximum[rank] #keep only the degree(j) nearest
        #maxi = nlargest(rank, dist[j])[-1]
        eps = 1e-10 #allowed error in the computation of the distances
        maxi += eps
        neighbors_list = [int(n*i) for n,i in enumerate(adjacency_matrix[j]) if int(i) !=0] #list of neighbors of j in G
        indices = [n for n,i in enumerate(dist[j]) if i <= maxi]
        indices.remove(j) #we don't want the node j to be itself in the list
        indices = [indices[i] for i in range(len(neighbors_list))] #we take exactly the degree(j) nearest nodes in the embedding
        D.append(1-float(len(set(indices)&set(neighbors_list)))/float(len(indices)))#dissimilarity of neighbors_list and indices
    print('Dissimilarity computed')
    return D

def structural_equivalence(adjacency_matrix, embedding, metric='euclidean'):
    N = len(adjacency_matrix) #number of nodes
    if N > 2000: 
        A = [i for i in range(N)]
        fraction_of_sampled_nodes = 0.1
        number_of_sampled_nodes = int(N*fraction_of_sampled_nodes)
        list_of_random_nodes = random.sample(A, number_of_sampled_nodes)
        #if there are more than 2k nodes, then we sample 10% of them
    else:
        #else we take all the nodes
        list_of_random_nodes = range(N)
    
    less_emb = []
    less_adjM = []
    
    N2 = len(list_of_random_nodes)
    for w in list_of_random_nodes:
        less_emb.append(embedding[w])
        less_adjM.append(adjacency_matrix[w])
    print('computing distances') #compute distances between vectors of the embedding space
    dist = embedding_distance(less_emb, metric)
    struc_dist = embedding_distance(less_adjM, metric) #compute distances between lines of the adj matrix
    struc_distances_list = []
    embedding_distances = []
    for i in range(N2):
        for j in range(N2):
            if i<j: #we don't take i=j since distances will be both 0, and d[i][j] = d[j][i]
                struc_distances_list.append(struc_dist[i][j])
                embedding_distances.append(dist[i][j])
    print('Structural equivalence computed')
    #return spearmanr(struc_distances_list, embedding_distances)
    return pearsonr(struc_distances_list, embedding_distances) #return the correlation coefficient 
    

def isomorphic_equivalence(embedding, isom_distances, list_of_sampled_nodes, metric='euclidean'):
    """This value is -value as compared to the paper """
    N = len(list_of_sampled_nodes)
    less_embedding = []
    for w in list_of_sampled_nodes:
        less_embedding.append(embedding[w])
    dist = embedding_distance(less_embedding, metric)
    isom_distances_list = []
    embedding_distances = []
    for i in range(N):
        for j in range(N):
            if i<j: #same as above
                isom_distances_list.append(isom_distances[i][j])
                embedding_distances.append(dist[i][j])
    print('isom equivalence computed')
    #return spearmanr(isom_distances_list, embedding_distances) #return the correlation coefficient
    return pearsonr(isom_distances_list, embedding_distances)
    



def graph_dist(G, radius_of_egonet, list_of_sampled_nodes):
    #compute pairwise ged between all pairs of subgraphs
    graph_list = []
    k=0
    print('Converting graph from networkx to gxl format')
    for i in list_of_sampled_nodes: #for every node
        A = nx.ego_graph(G,i,radius=radius_of_egonet, center=False, undirected= True, distance=None) #find its ego-network
        networkx_to_gxl(A, G.name + "_" + str(i+1), "./graph_matching_inputs/" + G.name + "_" + str(i+1) + ".gxl") #write it as a gxl format
        graph_list.append(G.name + "_" + str(i+1) + ".gxl") #don't forget to proceed it
        print('Conversion: '+repr(k/len(list_of_sampled_nodes)*100)+'%')
        k += 1
    gxl_to_xml(graph_list, "./graph_matching_inputs/" + G.name + ".xml") #write all the graphs you want to compare together
    write_gmt_prop(G, "./graph_matching_inputs/" + G.name + ".prop")     #write how you want to compare them
    order = "java -jar ./graph-matching-toolkit.jar " + "./graph_matching_inputs/" + G.name + ".prop" #compute the graph edit distances
    os.system(order)
    gdist = read_gmt_results("./graph_matching_outputs/results_"+ G.name + ".txt", len(list_of_sampled_nodes)) # read the ged
    return gdist


def comp_clusters_communities(embedding, labels_communities, algo = True, n_clusters = 5):
    X = StandardScaler().fit_transform(embedding) #rescaling of the data
    if algo: #choose which algo you want to find communities with
        db = DBSCAN().fit(X)
        labels_clusters = db.labels_
    else:
        kM = KMeans(n_clusters = n_clusters).fit(X)
        labels_clusters = kM.labels_
    return ami(labels_clusters, labels_communities) #adjusted mutual information between ground truth and communities discovered by the algorithm


def pca_adjacency_matrix(G, dimension):
    #performs pca on adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_matrix(G)
    pca = PCA(n_components = dimension)
    return pca.fit_transform(adjacency_matrix)


def svd_adjacency_matrix(G, dimension):
    #performs svd on adj matrix
    adjacency_matrix = nx.to_scipy_sparse_matrix(G)
    svd = TruncatedSVD(n_components = dimension)
    return svd.fit_transform(adjacency_matrix)


def ica_adjacency_matrix(G, dimension):
    #performs ica on adj matrix
    adjacency_matrix = nx.to_scipy_sparse_matrix(G)
    ica = FastICA(n_components = dimension)
    return ica.fit_transform(adjacency_matrix)


def nmf_adjacency_matrix(G, dimension):
    #performs nmf on adj matrix
    adjacency_matrix = nx.to_scipy_sparse_matrix(G)
    nmf = NMF(n_components = dimension)
    return nmf.fit_transform(adjacency_matrix)


def mds_shortest_paths(G, dimension, cutoff = None):
    #compute MDS on a specific matrix 
    SPL = G.graph['SPL'] #shortest path lengths
    A = DataFrame(SPL).fillna(100)  #open the shortest path lengths as a dataframe
    A.sort_index(axis = 1, inplace = True)
    A.sort_index(axis = 0, inplace = True)
    df = A.values #returns it as a np array whose element i,j is the shortest path length between node i and j
    for i in range(len(df)):
        for j in range(len(df)):
            if df[i][j] == 1: #if SPL = 1
                df[i][j] = 0 #set SPL to 0
            elif df[i][j] > 1: #if SPL > 1
                df[i][j] = 100 #set SPL to 100. Values can be changed according to what we want to study
    mds = MDS(n_components = dimension, metric = True, dissimilarity = 'precomputed')
    return mds.fit_transform(A.values) #performs MDS


def LE(G, dimension):
    #compute Laplacian Eigenmaps 
    norm_lap = nx.normalized_laplacian_matrix(G)
    u, v = sparse.linalg.eigs(norm_lap, k = dimension+1, which = 'SM')
    return v[:, 1:] 


def LLE(G, dimension):
    #compute Locally linear embedding
    A = nx.to_scipy_sparse_matrix(G) #adjacency matrix
    normalize(A, norm='l1', axis=1, copy=False)
    Id = sparse.eye(G.number_of_nodes())
    Diff = Id - A
    u, sigma, v_t = sparse.linalg.svds(Diff, k = dimension+1, which='SM')
    v = v_t.T
    return v[:, 1:]


def HOPE(G, dimension, alpha):
    #compute HOPE embedding
    A = nx.to_numpy_matrix(G)
    Id = np.eye(len(A))
    B = Id - alpha*A
    C = alpha*A
    D = np.dot(np.linalg.inv(B),C) #product of B^-1 and C
    u, sigma, v_t = sparse.linalg.svds(D, k = dimension//2)
    X1 = np.dot(u, np.diag(np.sqrt(sigma)))
    X2 = np.dot(v_t.T, np.diag(np.sqrt(sigma)))
    return np.concatenate((X1, X2), axis=1)


def node2vec(i='input_n2v', o='output_n2v', d=128, l=80, r=10, k=10, e=1, p=1, q=1):
    #compute node2vec embedding
    orders = './node2vec/node2vec ' + '-i:' + i + ' -o:' + o + ' -d:' + str(d) + ' -l:' + str(l) + ' -r:' + str(r) + ' -k:' + str(k) + ' -e:' + str(e) + ' -p:' + str(p) + ' -q:' + str(q)
    os.system(orders)
    return read_emb(o)









def load_graph(G, edgefile, name):
    G_2 = G.copy()
    G_2.name = name
    try:
        with open(edgefile): pass
    except:
        write_edgelist(G, edgefile)
    G_2.graph['edgelist'] = edgefile
    G_2.graph['bcsr'] = './verse_input/' + G_2.name + '.bcsr'
    G_2.graph['verse.output'] = './verse_output/' + G_2.name + '.bin'
    try:
        with open(G_2.graph['bcsr']): pass
    except:
        os.system('python ../verse-master/python/convert.py ' + G_2.graph['edgelist'] + ' ' + G_2.graph['bcsr'])
    return G_2

def read_emb(file_to_read):
    #read embedding file where first line is number of nodes, dimension of embedding and next lines are node_id, embedding vector
    with open(file_to_read, 'r') as f:
        number_of_nodes, dimension = f.readline().split()
        number_of_nodes = int(number_of_nodes)
        dimension = int(dimension)
        Y = [[0 for i in range(dimension)] for j in range(number_of_nodes)]
        for i, line in enumerate(f):
            line = line.split()
            Y[int(line[0])] = [float(line[j]) for j in range(1, dimension+1)]
    return Y



def write_emb(filename, embedding, dimension, start_time, end_time):
    #write the embedding as a .tsv file with start and end time
    with open(filename,'w') as myfile:
        myfile.write('#start time: '+str(time.asctime( time.localtime(start_time)))+'\n')
        myfile.write('#end time: '+str(time.asctime( time.localtime(end_time)))+'\n')
        myfile.write('#time for computation of the embedding: '+str(end_time-start_time)+'\n')
        for i in range(len(embedding)):
            string = str(embedding[i][0].real)
            for j in range(dimension-1):
                string = string + '\t'+str(embedding[i][j+1].real)
            string = string + '\n'
            myfile.write(string)
    return 0

def dict_to_emb(dicti):
    #od = collections.OrderedDict(sorted(dicti.items()))
    #emb = []
    #for k, v in od.items():
    #    emb.append(v)
    emb = [[] for i in range(len(dicti))]
    for i in range(len(dicti)):
        emb[i] = dicti[i]
    return emb            

def write_edgelist(G, filename):
    #writes edgelist for undirected unweighted graphs
    with open(filename, 'w') as myfile:
        for e in G.edges():
            line = str(e[0]) + ' ' + str(e[1]) + '\n'
            myfile.write(line)
    return 0

def read_edgelist(filename):
    #read edgelist for undirected unweighted graphs and return the corresponding graph
    G = nx.Graph()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                pass
            else:
                line = line.split()
                G.add_edge(int(line[0]), int(line[1]))
    return G

def write_results(filename, results, names):
    #write results of the measures for every embedding algo in 'names' in filename
    with open(filename, 'w') as myfile:
        for i in range(len(results)):
            string = names[i] + " "
            for j in range(len(results[i])):
                string += str(results[i][j]) + " "
            string += '\n'
            myfile.write(string)
    return 0


def read_Dancer_file(filename):
    #read dancer edgelist and communities
    labels_communities = []
    edgelist = []
    with open(filename, 'r') as myfile:
        for i, line in enumerate(myfile):
            if line.startswith('#'):
                continue
            words = line.split(';')
            if len(words) == 3:
                labels_communities.append(int(words[2]))
            if len(words) == 2:
                edgelist.append((int(words[0]), int(words[1])))
    return labels_communities, edgelist

def networkx_to_gxl(G, graphname, filename):
    #writes a graph in networkx format to gxl format
    with open(filename, 'w') as myfile:
        intro = "<?xml version=\"1.0\"?>" + "\n" + "<!DOCTYPE gxl SYSTEM \"http://www.gupro.de/GXL/gxl-1.0.dtd\">" + "\n"
        myfile.write(intro)
        if nx.is_directed(G):
            graph_prop = "<gxl><graph id=\"" + str(graphname) + "\" edgeids=\"false\" edgemode=\"defaultdirected\">"
        else:
            graph_prop = "<gxl><graph id=\"" + str(graphname) + "\" edgeids=\"false\" edgemode=\"defaultundirected\">"
        myfile.write(graph_prop)
        for i in G.nodes():
            node_prop = "<node id=\"_" + str(i+1) + "\"></node>"
            myfile.write(node_prop)
        for edge in G.edges():            
            edge_prop = "<edge from=\"_" + str(int(edge[0])+1) + "\" to=\"_" + str(int(edge[1])+1) + "\"></edge>"
            myfile.write(edge_prop)
        myfile.write("</graph></gxl>")

def gxl_to_xml(graph_list, filename):
    #write a collection of gxl graphs in xml format
    with open(filename, 'w') as myfile:
        myfile.write("<?xml version=\"1.0\"?>" + "<GraphCollection><graphs>")
        for graphname in graph_list:
            myfile.write("<print file=\""+ graphname + "\"/>")
        myfile.write("</graphs></GraphCollection>")
        

def read_gmt_results(filename, number_of_sampled_nodes):
    #read the results of the graph matching toolkit
    N = number_of_sampled_nodes
    results = [[0 for i in range(N)] for j in range(N)]
    k = 0
    with open(filename, 'r') as myfile:
        for i, line in enumerate(myfile):
            if line.startswith('#') or line.startswith('*') or line.startswith('\n') or line.startswith(' '):
                continue
            results[k%N][k//N] = float(line)
            k += 1
    return results
    
           
def write_gmt_prop(G, filename):
    #write property file for the graph matching toolkit
    with open(filename, 'w') as myfile:
        myfile.write("source=./graph_matching_inputs/" + G.name + ".xml\n")
        myfile.write("target=./graph_matching_inputs/" + G.name + ".xml\n")
        myfile.write("path=./graph_matching_inputs/\n")
        myfile.write("result=./graph_matching_outputs/results_" + G.name + ".txt\n")
        string = "matching=Beam\ns=100\nadj=best\nnode=1.0\nedge=1.0\nnumOfNodeAttr=0\nnodeAttr0=\nnodeCostType0=\nnodeAttr0Importance=\nmultiplyNodeCosts=0\npNode=1\nundirected=1\nnumOfEdgeAttr=0\nedgeAttr0=\nedgeCostType0=\nedgeAttr0Importance=\nmultiplyEdgeCosts=0\npEdge=1\nalpha=0.5\noutputGraphs=0\noutputEditpath=0\noutputCostMatrix=0\noutputMatching=0\nsimKernel=4"
        myfile.write(string)


def sampling(G, fraction_of_sampled_nodes):
    #samples a fraction of nodes of a graph
    N = G.number_of_nodes()
    A = list(G.nodes)
    number_of_sampled_nodes = int(N*fraction_of_sampled_nodes)
    list_of_random_nodes = random.sample(A, number_of_sampled_nodes)
    return list_of_random_nodes



