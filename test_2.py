distance_matrix = np.zeros((len(all_aligned_beads),len(all_aligned_beads)))

for i, b_i in enumerate(all_aligned_beads):

    i_mol = all_aligned_beads_mol_index[i]

    for j, b_j in enumerate(all_aligned_beads):

        j_mol = all_aligned_beads_mol_index[j]

        if i_mol == j_mol:

            distance_matrix[i,j] = 10000000000

            distance_matrix[j, i] = 10000000000

        elif i < j:

            d = np.linalg.norm(b_i - b_j)

            distance_matrix[i,j] = d

            distance_matrix[j,i] = d


from sklearn.cluster import AgglomerativeClustering

AggCluster = AgglomerativeClustering( distance_threshold=R*2 ,n_clusters=None ,affinity='precomputed', linkage='complete' )

cs = AggCluster.fit(distance_matrix)

