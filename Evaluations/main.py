import matplotlib.pyplot as plt
from evaluation import functions as fct
import time
import networkx as nx
from scipy.spatial.distance import cdist, jaccard
import numpy as np
import os
import csv
from sdne import SDNE

timo = time.time()

#import graphs. Format is edgelist. Graphs are undirected and unweighted.
graphs = []

edgefile = './gem/data/karate.edgelist'
G = fct.read_edgelist(edgefile)
G = fct.load_graph(G, edgefile, 'ZKC')
graphs.append(G)

#G = nx.generators.gnp_random_graph(100, 0.1, seed=42, directed=False)
#edgefile = "Gnp100.edgelist"
#G = fct.load_graph(G, edgefile, 'Gnp100')
#graphs.append(G)

#G = nx.generators.barabasi_albert_graph(100, 10, seed=42)
#edgefile = "BA100.edgelist"
#G = fct.load_graph(G, edgefile, 'BA100')
#graphs.append(G)

#edgefile = './gem/data/dancer100.edgeList'
#G = fct.read_edgelist(edgefile)
#G = fct.load_graph(G, edgefile, 'Dancer_100')
#graphs.append(G)

#edgefile = './gem/data/dancer1000.edgeList'
#G = fct.read_edgelist(edgefile)
#G = fct.load_graph(G, edgefile, 'Dancer_1k')
#graphs.append(G)

#edgefile = './gem/data/email.edgelist'
#G = fct.read_edgelist(edgefile)
#G = fct.load_graph(G, edgefile, 'email')
#mapping = {}
#for i in range(G.number_of_nodes()):
#    mapping[i+1] = i
#G = nx.relabel_nodes(G, mapping)
#graphs.append(G)

#edgefile = "Gnp1000.edgelist"
#G = nx.generators.gnp_random_graph(1000, 0.01, seed=42, directed=False)
#G = fct.load_graph(G, edgefile, 'Gnp1000')
#graphs.append(G)

#G = nx.generators.barabasi_albert_graph(1000, 10, seed=42)
#edgefile = "BA1000.edgelist"
#G = fct.load_graph(G, edgefile, 'BA1000')
#graphs.append(G)

#G = nx.generators.gnp_random_graph(10000, 0.001, seed=42, directed=False)
#edgefile = "Gnp10000.edgelist"
#G = fct.load_graph(G, edgefile, 'Gnp10000')
#graphs.append(G)

#edgefile = './gem/data/toto_0.edgeList'
#G = fct.read_edgelist(edgefile)
#G = fct.load_graph(G, edgefile, 'Dancer_10k')
#graphs.append(G)

#edgefile = './gem/data/PGP.edgelist'
#G = fct.read_edgelist(edgefile)
#G = fct.load_graph(G, edgefile, 'PGP')
#mapping = {}
#for i in range(G.number_of_nodes()):
#    mapping[i+1] = i
#G = nx.relabel_nodes(G, mapping)
#graphs.append(G)

#G = nx.generators.barabasi_albert_graph(10000, 10, seed=42)
#edgefile = "BA10k.edgelist"
#G = fct.load_graph(G, edgefile, 'BA10k')
#graphs.append(G)


for G in graphs:

   #first, we need to compute a few things: adj_matrix, graph edit distances
    
    A = nx.to_numpy_matrix(G)
    A = np.array(A)

    if G.number_of_nodes() < 2000:
        G.graph['SPL'] = dict(nx.all_pairs_shortest_path_length(G, cutoff = None))
        print('SPL computed')

    if G.number_of_nodes() < 200:
        fraction_of_sampled_nodes = 1
    elif 200 < G.number_of_nodes() < 5000:
        fraction_of_sampled_nodes = 0.1
    else:
        fraction_of_sampled_nodes = 0.01
    list_of_sampled_nodes = fct.sampling(G, fraction_of_sampled_nodes)
    neighborhood_size = 1
    ged = fct.graph_dist(G, neighborhood_size, list_of_sampled_nodes)
    print('GED succesfully computed')
    print('Current time: '+ repr(time.time()-timo))

    dim = []
    
    mean_LE = []
    mean2_LE = []
    mean3_LE = []
    
    mean_LLE = []
    mean2_LLE = []
    mean3_LLE = []
    
    mean_HOPE = []
    mean2_HOPE = []
    mean3_HOPE = []

    mean_n2vA = []
    mean2_n2vA = []
    mean3_n2vA = []
    
    mean_n2vB = []
    mean2_n2vB = []
    mean3_n2vB = []
    
    mean_sdne = []
    mean2_sdne = []
    mean3_sdne = []
    
    mean_s2v = []
    mean2_s2v = []
    mean3_s2v = []
    
    mean_spring = []
    mean2_spring = []
    mean3_spring = []

    mean_svd = []
    mean2_svd = []
    mean3_svd = []
    
    mean_mds = []
    mean2_mds = []
    mean3_mds = []

    mean_verse = []
    mean2_verse = []
    mean3_verse = []

    #choose dimension of the embedding
    #DIMENSION = [2]
    if G.number_of_nodes() < 200:
        DIMENSION = range(2,G.number_of_nodes()-2)
    elif 200 < G.number_of_nodes() < 5000:
        DIMENSION = [2, 10, 50, 100, 200, 500, G.number_of_nodes()-5]
    else:
        DIMENSION = [2, 10, 100, 500, 1000]

    #choose which embedding you want to compute
    for dimension in DIMENSION:
        print(G.name, dimension)
        print('number of nodes: ', G.number_of_nodes())
        dim.append(dimension)

        print('doing LE '+str(dimension))
        try:
            start_time = time.time() #starting time of the computation of the embedding
            Y = fct.LE(G, dimension) #compute the embedding
            end_time = time.time()   #end time of the computation
            filename = './embeddings/LE_' + G.name + '_' + str(dimension)+'.txt' #where to write the results
            fct.write_emb(filename, Y, dimension, start_time, end_time) #write the embedding, the start and end times
            D = fct.neighbors_dissimilarity(A, Y, metric = 'euclidean') #compute the dissimilarity as described in the paper
            D2 = fct.structural_equivalence(A, Y, metric = 'euclidean') #compute structural equivalence
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'euclidean') #compute isomorphic equivalence
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_LE.append(np.mean(D))
        mean2_LE.append(np.mean(D2))
        mean3_LE.append(np.mean(D3))
        print('LE computed')

        print('doing LLE: '+str(dimension))
        try:
            start_time = time.time()
            Y = fct.LLE(G, dimension)
            end_time = time.time()
            filename = './embeddings/LLE_' + G.name + '_' + str(dimension)+'.txt'
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'euclidean')
            D2 = fct.structural_equivalence(A, Y, metric = 'euclidean')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'euclidean')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_LLE.append(np.mean(D))
        mean2_LLE.append(np.mean(D2))
        mean3_LLE.append(np.mean(D3))
        print('LLE computed')


        if G.number_of_nodes() < 2000:
            print('doing mds:' + str(dimension))
            try:
                start_time = time.time()
                Y = fct.mds_shortest_paths(G, dimension)
                end_time = time.time()
                filename = './embeddings/MDS_' + G.name + '_' + str(dimension)+'.txt'
                fct.write_emb(filename, Y, dimension, start_time, end_time)
                D = fct.neighbors_dissimilarity(A, Y, metric = 'euclidean')
                D2 = fct.structural_equivalence(A, Y, metric = 'euclidean')
                D2 = D2[0]
                D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'euclidean')
                D3 = D3[0]
            except:
                D = [100]
                D2 = [100]
                D3 = [100]
            mean_mds.append(np.mean(D))
            mean2_mds.append(np.mean(D2))
            mean3_mds.append(np.mean(D3))
            print('MDS computed')

        print('doing HOPE: '+str(dimension))
        try:
            start_time = time.time()
            Y = fct.HOPE(G, dimension, 0.01)
            end_time = time.time()
            filename = './embeddings/HOPE_' + G.name + '_' + str(dimension)+'.txt'
            if dimension%2 == 0:
                fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_HOPE.append(np.mean(D))
        mean2_HOPE.append(np.mean(D2))
        mean3_HOPE.append(np.mean(D3))
        print('HOPE computed')

        print('doing n2vA: '+str(dimension))
        try:
            start_time = time.time()
            Y = fct.node2vec(i=G.graph['edgelist'], d=dimension, l=80, r=10, k=10, e=1, p=0.5, q=4)
            end_time = time.time()
            filename = './embeddings/n2vA_' + G.name + '_' + str(dimension)+'.txt'
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_n2vA.append(np.mean(D))
        mean2_n2vA.append(np.mean(D2))
        mean3_n2vA.append(np.mean(D3))

        print('doing n2vB: '+str(dimension))
        try:
            start_time = time.time()
            Y = fct.node2vec(i=G.graph['edgelist'], d=dimension, l=80, r=10, k=10, e=1, p=4, q=0.5)
            end_time = time.time()
            filename = './embeddings/n2vB_' + G.name + '_' + str(dimension)+'.txt'
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_n2vB.append(np.mean(D))
        mean2_n2vB.append(np.mean(D2))
        mean3_n2vB.append(np.mean(D3))


        if G.number_of_nodes() < 200:
            n_units = [15, 3,]
            n_batch = int(G.number_of_nodes()/10)
        else:
            n_units = [50, 15,]
            n_batch = int(G.number_of_nodes()/50)
        print('doing SDNE: '+str(dimension))
        method = SDNE(d=dimension, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=1, n_units=n_units, rho=0.3, n_iter=3, xeta=0.01, n_batch=n_batch, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])
        try:
            start_time = time.time()
            Y, t = method.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            end_time = time.time()
            filename = './embeddings/SDNE_' + G.name + '_' + str(dimension)+'.txt'
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'euclidean')
            D2 = fct.structural_equivalence(A, Y, metric = 'euclidean')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'euclidean')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_sdne.append(np.mean(D))
        mean2_sdne.append(np.mean(D2))
        mean3_sdne.append(np.mean(D3))
        print('SDNE computed')

        print('doing s2v: '+str(dimension))
        try:
            orders = "python ../struc2vec-master/src/main.py --input " + G.graph['edgelist'] + " --output embedding.emb --num-walks 10 --walk-length 80 --window-size 10 --dimensions " + str(dimension) + " --OPT1 True --OPT2 True --OPT3 True --until-layer 6 --workers 8"
            start_time = time.time()
            os.system(orders)
            end_time = time.time()
            filename = './embeddings/s2v_' + G.name + '_' + str(dimension)+'.txt'
            Y = fct.read_emb('embedding.emb') 
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_s2v.append(np.mean(D))
        mean2_s2v.append(np.mean(D2))
        mean3_s2v.append(np.mean(D3))
        print('s2v computed')

        print('doing verse: ' + str(dimension))
        try:
            start_time = time.time()
            orders = "../verse-master/src/verse -input " + G.graph['bcsr'] + " -output " + G.graph['verse.output'] + " -dim " + str(dimension) + " -alpha 0.85 -threads 8 -nsamples 3"
            os.system(orders)
            end_time = time.time()
            filename = './embeddings/verse_' + G.name + '_' + str(dimension)+'.txt'
            Y = np.fromfile(G.graph['verse.output'], np.float32).reshape(G.number_of_nodes(), dimension)
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_verse.append(np.mean(D))
        mean2_verse.append(np.mean(D2))
        mean3_verse.append(np.mean(D3))
        print('verse computed')


        if G.number_of_nodes() < 200:
            print('doing KKL: '+str(dimension))
            try:
                start_time = time.time()
                Y = nx.drawing.layout.kamada_kawai_layout(G, dim=dimension)#, iterations=500)
                #if using networkx layouts, you need to convert the dictionary to the usual embedding format
                end_time = time.time()
                filename = './embeddings/KKL_' + G.name + '_' + str(dimension)+'.txt'
                Y = fct.dict_to_emb(Y)
                fct.write_emb(filename, Y, dimension, start_time, end_time)
                D = fct.neighbors_dissimilarity(A, Y, metric = 'euclidean')
                D2 = fct.structural_equivalence(A, Y, metric = 'euclidean')
                D2 = D2[0]
                D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'euclidean')
                D3 = D3[0]
            except:
                D = [100]
                D2 = [100]
                D3 = [100]
            mean_spring.append(np.mean(D))
            mean2_spring.append(np.mean(D2))
            mean3_spring.append(np.mean(D3))
            print('KKL computed')

        print('doing svd: '+ str(dimension))
        try:
            start_time = time.time()
            Y = fct.svd_adjacency_matrix(G, dimension)
            end_time = time.time()
            filename = './embeddings/svd_' + G.name + '_' + str(dimension)+'.txt'
            fct.write_emb(filename, Y, dimension, start_time, end_time)
            D = fct.neighbors_dissimilarity(A, Y, metric = 'cosine')
            D2 = fct.structural_equivalence(A, Y, metric = 'cosine')
            D2 = D2[0]
            D3 = fct.isomorphic_equivalence(Y, ged, list_of_sampled_nodes, metric = 'cosine')
            D3 = D3[0]
        except:
            D = [100]
            D2 = [100]
            D3 = [100]
        mean_svd.append(np.mean(D))
        mean2_svd.append(np.mean(D2))
        mean3_svd.append(np.mean(D3))
        print('svd computed')

        print('\n\n\nCURRENT TIME:', time.time()-timo)
        print('\n\n')

    #plot results. Dissimilarity or structural equivalence or isomorphic equivalence vs dimension and write them too.
    plt.figure()
    plt.plot(dim, mean_LE, 'b+')
    plt.plot(dim, mean_LLE, 'r+')
    plt.plot(dim, mean_HOPE, 'g+')
    plt.plot(dim, mean_s2v, 'k+')
    if G.number_of_nodes() < 200:
        plt.plot(dim, mean_spring, 'c+')
    plt.plot(dim, mean_n2vA, 'm+')
    plt.plot(dim, mean_n2vB, 'y+')
    plt.plot(dim, mean_sdne, color = 'xkcd:grey', marker = '+', linestyle = 'None')
    plt.plot(dim, mean_svd, color = 'xkcd:bright green', marker = '+', linestyle = 'None')
    if G.number_of_nodes() < 2000:
        plt.plot(dim, mean_mds, color = 'xkcd:hot pink', marker = '+', linestyle = 'None')
    plt.plot(dim, mean_verse, color = 'xkcd:light brown', marker = '+', linestyle = 'None')

    title = 'Neighbors_dissimilarity_'+G.name
    plt.title(title)
    plt.xlabel('Dimension')
    if G.number_of_nodes() > 500:
        plt.xscale('log')
    plt.ylabel('Neighbors dissimilarity')
    if G.number_of_nodes() > 2000:
        plt.axis([0, 1050, -0.1, 1.1])
    else:
        plt.axis([0, len(A), -0.1, 1.1])
    #plt.legend(('LE', 'LLE', 'HOPE', 's2v', 'KKL', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'mds', 'verse'))
    plt.savefig('../Figures/Neigh_dissim/'+title)
    if G.number_of_nodes() < 101:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean_LE, mean_LLE, mean_HOPE, mean_s2v, mean_n2vA, mean_n2vB, mean_sdne, mean_svd, mean_verse, mean_spring, mean_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'KKL', 'mds'])
    elif 101 < G.number_of_nodes() < 1050:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean_LE, mean_LLE, mean_HOPE, mean_s2v, mean_n2vA, mean_n2vB, mean_sdne, mean_svd, mean_verse, mean_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'mds'])
    else:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean_LE, mean_LLE, mean_HOPE, mean_s2v, mean_n2vA, mean_n2vB, mean_sdne, mean_svd, mean_verse],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse'])


    plt.figure()
    plt.plot(dim, mean2_LE, 'b+')
    plt.plot(dim, mean2_LLE, 'r+')
    plt.plot(dim, mean2_HOPE, 'g+')
    plt.plot(dim, mean2_s2v, 'k+')
    if G.number_of_nodes() < 200:
        plt.plot(dim, mean2_spring, 'c+')
    plt.plot(dim, mean2_n2vA, 'm+')
    plt.plot(dim, mean2_n2vB, 'y+')
    plt.plot(dim, mean2_sdne, color = 'xkcd:grey', marker = '+', linestyle = 'None')
    plt.plot(dim, mean2_svd, color = 'xkcd:bright green', marker = '+', linestyle = 'None')
    if G.number_of_nodes() < 2000:
        plt.plot(dim, mean2_mds, color = 'xkcd:hot pink', marker = '+', linestyle = 'None')
    plt.plot(dim, mean2_verse, color = 'xkcd:light brown', marker = '+', linestyle = 'None')
    
    title = 'Structural_equivalence_'+G.name
    plt.title(title)
    plt.xlabel('Dimension')
    if G.number_of_nodes() > 500:
        plt.xscale('log')
    plt.ylabel('Pearson coefficient')
    if G.number_of_nodes() > 2000:
        plt.axis([0, 1050, -1.1, 1.1])
    else:
        plt.axis([0, len(A), -1.1, 1.1])
#    plt.legend(('LE', 'LLE', 'HOPE', 's2v', 'spring', 'n2v_p05_q4', 'n2v_p4_q05', 'svd', 'mds', 'verse'))
    plt.savefig('../Figures/Structural_eq/'+title)

    if G.number_of_nodes() < 101:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean2_LE, mean2_LLE, mean2_HOPE, mean2_s2v, mean2_n2vA, mean2_n2vB, mean2_sdne, mean2_svd, mean2_verse, mean2_spring, mean2_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'KKL', 'mds'])
    elif 101 < G.number_of_nodes() < 1050:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean2_LE, mean2_LLE, mean2_HOPE, mean2_s2v, mean2_n2vA, mean2_n2vB, mean2_sdne, mean2_svd, mean2_verse, mean2_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'mds'])
    else:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean2_LE, mean2_LLE, mean2_HOPE, mean2_s2v, mean2_n2vA, mean2_n2vB, mean2_sdne, mean2_svd, mean2_verse],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse'])


    
    plt.figure()
    plt.plot(dim, mean3_LE, 'b+')
    plt.plot(dim, mean3_LLE, 'r+')
    plt.plot(dim, mean3_HOPE, 'g+')
    plt.plot(dim, mean3_s2v, 'k+')
    if G.number_of_nodes() < 200:
        plt.plot(dim, mean3_spring, 'c+')
    plt.plot(dim, mean3_n2vA, 'm+')
    plt.plot(dim, mean3_n2vB, 'y+')
    plt.plot(dim, mean3_sdne, color = 'xkcd:grey', marker = '+', linestyle = 'None')
    plt.plot(dim, mean3_svd, color = 'xkcd:bright green', marker = '+', linestyle = 'None')
    if G.number_of_nodes() < 2000:
        plt.plot(dim, mean3_mds, color = 'xkcd:hot pink', marker = '+', linestyle = 'None')
    plt.plot(dim, mean3_verse, color = 'xkcd:light brown', marker = '+', linestyle = 'None')

    title = 'Isomorphic_equivalence_'+G.name
    plt.title(title)
    plt.xlabel('Dimension')
    if G.number_of_nodes() > 500:
        plt.xscale('log')
    plt.ylabel('Pearson coefficient')
    if G.number_of_nodes() > 2000:
        plt.axis([0, 1050, -1.1, 1.1])
    else:
        plt.axis([0, len(A), -1.1, 1.1])
    #plt.legend(('LE', 'LLE', 'HOPE', 's2v', 'n2vA', 'n2vB', 'SDNE',  'svd', 'verse'))
    plt.savefig('../Figures/Isomorphic_equivalence/'+title)
    if G.number_of_nodes() < 101:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean3_LE, mean3_LLE, mean3_HOPE, mean3_s2v, mean3_n2vA, mean3_n2vB, mean3_sdne, mean3_svd, mean3_verse, mean3_spring, mean3_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'KKL', 'mds'])
    elif 101 < G.number_of_nodes() < 1050:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean3_LE, mean3_LLE, mean3_HOPE, mean3_s2v, mean3_n2vA, mean3_n2vB, mean3_sdne, mean3_svd, mean3_verse, mean3_mds],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse', 'mds'])
    else:
        fct.write_results('./Results/'+title, results = [DIMENSION, mean3_LE, mean3_LLE, mean3_HOPE, mean3_s2v, mean3_n2vA, mean3_n2vB, mean3_sdne, mean3_svd, mean3_verse],
                  names = ['dimension', 'LE', 'LLE', 'HOPE', 's2v', 'n2v_p05_q4', 'n2v_p4_q05', 'sdne', 'svd', 'verse'])

print(time.time()-timo)
plt.show()
