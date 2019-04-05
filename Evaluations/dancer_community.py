import matplotlib.pyplot as plt
from src import functions as fct
from sklearn.metrics import adjusted_mutual_info_score as ami
from time import time
import infomap
from sdne.sdne import SDNE
from igraph import Graph as ig
import graph_tool.all as gt
import community
import networkx as nx
import numpy as np
import os



timo = time()

# Load graphs
graphs = []

isDirected = False
mean_LE = []
stdDev_LE = []
mean_LLE = []
stdDev_LLE = []
mean_HOPE = []
stdDev_HOPE = []
mean_n2v_p05_q4 = []
stdDev_n2v_p05_q4 = []
mean_n2v_p4_q05 = []
stdDev_n2v_p4_q05 = []
mean_sdne = []
stdDev_sdne = []
mean_s2v = []
stdDev_s2v = []
mean_spring = []
stdDev_spring = []
mean_svd = []
stdDev_svd = []
mean_mds = []
stdDev_mds = []
mean_verse = []
stdDev_verse = []
mean_sbm = []
stdDev_sbm = []
mean_infomap = []
stdDev_infomap = []
mean_maxmod = []
stdDev_maxmod = []
p_ins = []
graph_mod = []

number_of_graphs = 20
graph_nums = [i*2 for i in range(number_of_graphs)]
for graph_num in graph_nums:
    graphs = []

    filename = './src/data/community_dancer2/t'+str(graph_num)+'.graph'
    ground_truth_comm, dancer_edgelist = fct.read_Dancer_file(filename)
    G = nx.Graph()
    for e in dancer_edgelist:
        G.add_edge(e[0], e[1])
    n = G.number_of_nodes()
    G.name = 'Dancer_community2_' + str(graph_num)
    G.graph['edgelist'] = './src/data/'+G.name+'.edgelist'
    fct.write_edgelist(G, G.graph['edgelist'])
    G.graph['bcsr'] = './verse_input/' + G.name + '.bcsr'
    os.system('python ../verse-master/python/convert.py ' + G.graph['edgelist'] + ' ' + G.graph['bcsr'])
    G.graph['verse.output'] = './verse_output/' + G.name + '.bin'
    G.graph['labels_communities'] = ground_truth_comm
    G.graph['number_communities'] = max(G.graph['labels_communities'])+1
    n_commu = G.graph['number_communities']
    communities = [set() for i in range(n_commu)]
    for i in range(n):
        for j in range(n_commu):
            if ground_truth_comm[i] == j:
                communities[j].add(i)
    graph_modularity = nx.algorithms.community.quality.modularity(G, communities)
    print("Modularity of "+G.name+": "+repr(graph_modularity))
    graph_mod.append(graph_modularity)
    p_ins.append(graph_modularity)
    #cluster_coeff = nx.algorithms.cluster.average_clustering(G)
    #print("Clustering coeff "+G.name+": "+repr(cluster_coeff))
    graphs.append(G)


    
    print('\n')
        
    zLE = []
    zLLE = []
    zHOPE = []
    zn2v_p05_q4 = []
    zn2v_p4_q05 = []
    zsdne = []
    zs2v = []
    zspring = []
    zsvd = []
    zmds = []
    zverse = []
    zsbm = []
    zinfomap = []
    zmaxmod = []
    


    
    for G in graphs:
        print(G.name)
        G.graph['SPL'] = dict(nx.all_pairs_shortest_path_length(G, cutoff = None))  
        A = nx.to_numpy_matrix(G)
        A = np.array(A)

        dimension = 2
            
        print("Doing LE") 
        timi = time()
        try:
            Y = fct.LE(G, dimension)
            Y = [[Y[i][j].real for j in range(len(Y[i]))] for i in range(len(Y))]
            D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        except:
            print("LE failed")
            D = [100]
        zLE.append(np.mean(D))
        print(time()-timi)
            
    
        print("Doing LLE")
        try:
            Y = fct.LLE(G, dimension)
            D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        except:
            print("LLE failed")
            D = [100]
        zLLE.append(np.mean(D))
        
    
        print("Doing HOPE")  
        Y = fct.HOPE(G, dimension, 0.01)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zHOPE.append(np.mean(D))
        
    
        print("Doing n2vA")
        Y = fct.node2vec(i=G.graph['edgelist'], d=dimension, l=80, r=10, k=10, e=1, p=0.5, q=4)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zn2v_p05_q4.append(np.mean(D))
        
    
        print("Doing n2vB")
        Y = fct.node2vec(i=G.graph['edgelist'], d=dimension, l=80, r=10, k=10, e=1, p=4, q=0.5)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zn2v_p4_q05.append(np.mean(D))
        
    
        method = SDNE(d=dimension, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=1,n_units=[50, 15,], rho=0.3, n_iter=3, xeta=0.01, n_batch=int(len(A)/50), modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])
        Y, t = method.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zsdne.append(np.mean(D))

        
    
        orders = "python ../struc2vec-master/src/main.py --input " + G.graph['edgelist'] + " --output embedding.emb --num-walks 10 --walk-length 80 --window-size 10 --dimensions " + str(dimension) + " --OPT1 True --OPT2 True --OPT3 True --until-layer 6 --workers 8"
        os.system(orders)
        Y = fct.read_emb('embedding.emb') #returns the embedding as a dictionary
        Y = fct.dict_to_emb(Y)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zs2v.append(np.mean(D))
    
        orders = "../verse-master/src/verse -input " + G.graph['bcsr'] + " -output " + G.graph['verse.output'] + " -dim " + str(dimension) + " -alpha 0.85 -threads 8 -nsamples 3"
        os.system(orders)
        Y = np.fromfile(G.graph['verse.output'], np.float32).reshape(G.number_of_nodes(), dimension)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zverse.append(np.mean(D))
        
        print("Doing spring")
        Y = nx.drawing.layout.kamada_kawai_layout(G, dim=dimension)#, iterations=500)
        #if using spring_layout, you need to convert the dictionary to the usual embedding format
        Y = fct.dict_to_emb(Y)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zspring.append(np.mean(D))
           
    
        try:
            Y = fct.svd_adjacency_matrix(A, dimension)
            D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        except:
            D = [100]
        zsvd.append(np.mean(D))
        
    
    
        Y = fct.mds_shortest_paths(G, dimension)
        D = fct.comp_clusters_communities(Y, G.graph['labels_communities'], algo = False, n_clusters = G.graph['number_communities'])
        zmds.append(np.mean(D))
        
    
    
        g = gt.load_graph_from_csv(G.graph['edgelist'], directed = isDirected, csv_options = {"delimiter": " ", "quotechar": '"'})
        block = gt.minimize_nested_blockmodel_dl(g, B_min = G.graph['number_communities'], B_max = G.graph['number_communities'])
        num_block = block.levels[0].get_B()
        block = block.levels[0].get_blocks()
        partition = [0 for i in range(G.number_of_nodes())]
        for i in range(G.number_of_nodes()): #for every node
            partition[i] = block[i]
        zsbm.append(ami(partition, G.graph['labels_communities']))
    
    
        igraph = ig.Read_Edgelist(G.graph['edgelist'])
        part = igraph.community_infomap()
        partition = [0 for i in range(G.number_of_nodes())]
        for i in range(G.number_of_nodes()):
            for j in range(len(part)):
                if i in part[j]:
                    partition[i] = j
        zinfomap.append(ami(partition, G.graph['labels_communities']))
            
    
        Y = community.best_partition(G.to_undirected()) #https://perso.crans.org/aynaud/communities/api.html
        #uses Louvain heuristices
        partition = [0 for i in range(G.number_of_nodes())]
        for k in range(G.number_of_nodes()):
            partition[k] = Y[k]
        zmaxmod.append(ami(partition, G.graph['labels_communities']))


        
    mean_LE.append(np.mean(zLE))
    stdDev_LE.append(np.std(zLE))
    
    mean_LLE.append(np.mean(zLLE))
    stdDev_LLE.append(np.std(zLLE))
    
    mean_HOPE.append(np.mean(zHOPE))
    stdDev_HOPE.append(np.std(zHOPE))
    
    mean_n2v_p05_q4.append(np.mean(zn2v_p05_q4))
    stdDev_n2v_p05_q4.append(np.std(zn2v_p05_q4))
    
    mean_n2v_p4_q05.append(np.mean(zn2v_p4_q05))
    stdDev_n2v_p4_q05.append(np.std(zn2v_p4_q05))
    
    mean_sdne.append(np.mean(zsdne))
    stdDev_sdne.append(np.std(zsdne))
    
    mean_s2v.append(np.mean(zs2v))
    stdDev_s2v.append(np.std(zs2v))
    
    mean_spring.append(np.mean(zspring))
    stdDev_spring.append(np.std(zspring))
    
    mean_svd.append(np.mean(zsvd))
    stdDev_svd.append(np.std(zsvd))
    
    mean_mds.append(np.mean(zmds))
    stdDev_mds.append(np.std(zmds))
    
    mean_verse.append(np.mean(zverse))
    stdDev_verse.append(np.std(zverse))
    
    mean_sbm.append(np.mean(zsbm))
    stdDev_sbm.append(np.std(zsbm))
    
    mean_infomap.append(np.mean(zinfomap))
    stdDev_infomap.append(np.std(zinfomap))
    
    mean_maxmod.append(np.mean(zmaxmod))
    stdDev_maxmod.append(np.std(zmaxmod))


plt.figure()
plt.errorbar(p_ins, mean_LE, yerr=stdDev_LE, color ='b', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_LLE, yerr=stdDev_LLE, color = 'r', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_HOPE, yerr=stdDev_HOPE, color = 'g', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_s2v, yerr=stdDev_s2v, color = 'k', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_spring, yerr=stdDev_spring, color = 'c', marker = '+', linestyle = ' ')
plt.errorbar(p_ins, mean_n2v_p05_q4, yerr=stdDev_n2v_p05_q4, color = 'm', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_n2v_p4_q05, yerr=stdDev_n2v_p4_q05, color = 'y', marker = '+', linestyle = '-')

title = 'Communities_Dancer_dim'+str(dimension)+'_A'
fct.write_results('./Results/Communities/'+title+'_mean', results = [p_ins, mean_LE, mean_LLE, mean_HOPE, mean_s2v, mean_spring, mean_n2v_p05_q4, mean_n2v_p4_q05],
                  names = ['p_in', 'LE', 'LLE', 'HOPE', 's2v', 'spring', 'n2v_p05_q4', 'n2v_p4_q05'])
fct.write_results('./Results/Communities/'+title+'_stdDev', results = [p_ins, stdDev_LE, stdDev_LLE, stdDev_HOPE, stdDev_s2v, stdDev_spring, stdDev_n2v_p05_q4, stdDev_n2v_p4_q05],
                  names = ['p_in', 'LE', 'LLE', 'HOPE', 's2v', 'spring', 'n2v_p05_q4', 'n2v_p4_q05'])
plt.title(title)
plt.xlabel('p_in')
plt.ylabel('Adj. Mutual Information')
plt.axis([0, 1.05, -0.1, 1.1])
plt.legend(('LE', 'LLE', 'HOPE', 's2v', 'spring', 'n2v_p05_q4', 'n2v_p4_q05'))
plt.savefig('../Figures/Communities/'+title)

plt.figure()
plt.errorbar(p_ins, mean_sdne, color = 'xkcd:light eggplant', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_svd, yerr=stdDev_svd, color = 'xkcd:bright green', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_mds, yerr=stdDev_mds, color = 'xkcd:dark purple', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_verse, yerr=stdDev_verse, color = 'xkcd:light brown', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_sbm, yerr=stdDev_sbm, color = 'xkcd:lilac', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_maxmod, yerr=stdDev_maxmod, color = 'xkcd:gold', marker = '+', linestyle = '-')
plt.errorbar(p_ins, mean_infomap, yerr=stdDev_infomap, color = 'xkcd:coral', marker = '+', linestyle = '-')


title = 'Communities_Dancer_dim'+str(dimension)+'_B'
fct.write_results('./Results/Communities/'+title+'_mean', results = [p_ins, mean_sdne, mean_svd, mean_mds, mean_verse, mean_sbm, mean_maxmod, mean_infomap],
                  names = ['p_in', 'sdne', 'svd', 'mds', 'verse', 'sbm', 'maxmod', 'infomap'])
fct.write_results('./Results/Communities/'+title+'_stdDev', results = [p_ins, stdDev_sdne, stdDev_svd, stdDev_mds, stdDev_verse, stdDev_sbm, stdDev_maxmod, stdDev_infomap],
                  names = ['p_in', 'sdne', 'svd', 'mds', 'verse', 'sbm', 'maxmod', 'infomap'])
plt.title(title)
plt.xlabel('p_in')
plt.ylabel('Adj. Mutual Information')
plt.axis([0, 1.05, -0.1, 1.1])
plt.legend(('sdne', 'svd', 'mds', 'verse', 'sbm', 'maxmod', 'infomap'))
plt.savefig('../Figures/Communities/'+title)

print(time()-timo)
