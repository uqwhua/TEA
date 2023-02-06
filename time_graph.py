def _build_dict_by_first_element(arrs):
    dic = {}
    for arr in arrs:
        if arr[0] not in dic:
            dic[arr[0]] = [arr]
        else:
            dic[arr[0]].append(arr)
    return dic


def _build_dict_by_list(elements, start_idx):
    dic = {}
    idx = start_idx
    for ele in elements:
        tup = tuple(ele)
        # assert tup not in dic
        # duplicate after padding
        if tup not in dic:
            dic[tup] = idx
            idx += 1
    return dic


def build_entity_and_value_edges(num_ent, attribute_triples):
    edges = []
    idx = num_ent
    for att_triple in attribute_triples:
        ent, val, att = att_triple
        edge = (idx, ent)
        edges.append(edge)
        idx = idx + 1
    return edges


def build_attr_edges(attr_triples, cluster, triple2id_dict, time_id2value, window=3):
    edges = []
    if len(attr_triples) < 2:
        return edges, []

    def get_time_value(x):
        return time_id2value[x[1]]

    attr_triples.sort(key=get_time_value, reverse=False)
    for i in range(len(attr_triples)):
        for j in range(i + 1, len(attr_triples)):
            attr_i = attr_triples[i][2]  # triple in format: entity, value, attr
            attr_j = attr_triples[j][2]

            if attr_i == attr_j:
                edges.append([tuple(attr_triples[i]), tuple(attr_triples[j])])
            elif attr_i in cluster and attr_j in cluster:
                edges.append([tuple(attr_triples[i]), tuple(attr_triples[j])])
            if len(edges) > window:
                break
        if len(edges) > window:
            break

    id_edges = []
    attr_edges = []
    for u, v in edges:
        u_id = triple2id_dict[u]
        u_attr = u[2]
        u_time_val = time_id2value[u[1]]
        v_id = triple2id_dict[v]
        v_attr = v[2]
        v_time_val = time_id2value[v[1]]

        if u_time_val > v_time_val:  # make directed graph
            id_edges.append([v_id, u_id])
            attr_edges.append([v_attr, u_attr])
        else:
            id_edges.append([u_id, v_id])
            attr_edges.append([u_attr, v_attr])

    return id_edges, attr_edges


def build_value_edges_in_cluster(num_ent, time_id2value, attribute_triples, clusters):
    # dic={ent1:[triple1,triple2,triple3],ent2:[]...}
    ent2triple_dict = _build_dict_by_first_element(attribute_triples)
    # attr node id starts from num_ent
    triple2id_dict = _build_dict_by_list(attribute_triples, num_ent)

    val_val_edges = []  # edges between values
    ent_val_edges = []  # edge connects val and ent
    for ent, attr_triples in ent2triple_dict.items():
        for cluster in clusters:
            id_edges, attr_edges = build_attr_edges(attr_triples, cluster, triple2id_dict, time_id2value)
            val_val_edges = val_val_edges + id_edges
            ent_val_edges = ent_val_edges + attr_edges
    if len(val_val_edges) > 0 and len(ent_val_edges) > 0:
        return val_val_edges, ent_val_edges
    else:  # to avoid None
        return [[num_ent, num_ent + 1]], [[0, 1]]


def build_dynamic_value_edges_in_cluster(num_graph, num_ent, time_id2value, attribute_triples, clusters):
    # dynamic graph
    val_edge_list = []
    val_edge_attr_list = []

    # dic={ent1:[triple1,triple2,triple3],ent2:[]...}
    ent2triple_dict = _build_dict_by_first_element(attribute_triples)
    # attr node id starts from num_ent
    triple2id_dict = _build_dict_by_list(attribute_triples, num_ent)
    for i in range(num_graph):
        graph_i = []
        val_val_edges = []  # edges between values
        val_edge_attrs = []  # edge connects val and ent
        for ent, attr_triples in ent2triple_dict.items():
            for cluster in clusters:
                id_edges, attr_edges = build_attr_edges(attr_triples, cluster, triple2id_dict,
                                                        time_id2value, window=i)
                val_val_edges = val_val_edges + id_edges
                val_edge_attrs = val_edge_attrs + attr_edges
        # avoid None
        if len(val_val_edges) == 0 or len(val_edge_attrs) == 0:
            val_val_edges = [[num_ent, num_ent + 1]]
            val_edge_attrs = [[0, 1]]

        val_edge_list.append(val_val_edges)
        val_edge_attr_list.append(val_edge_attrs)
    return val_edge_list, val_edge_attr_list


def split_by_clusters(attr_triples, clusters):
    splited = []

    def get_covered_triples(attr_triples, cluster):
        if len(attr_triples) < 2:
            return []
        covered = []
        for i in range(len(attr_triples) - 1):
            for j in range(i + 1, len(attr_triples)):
                attr_i = attr_triples[i][2]
                attr_j = attr_triples[j][2]
                if attr_i == attr_j:
                    covered.append([attr_triples[i], attr_triples[j]])
                elif attr_i in cluster and attr_j in cluster:
                    covered.append([attr_triples[i], attr_triples[j]])

        return covered

    for cluster in clusters:
        split_i = get_covered_triples(attr_triples, cluster)
        if len(split_i) > 0:
            splited = splited + split_i

    return splited


def build_dynamic_time_graphs(num_ent,time_id2value, attribute_triples, clusters):
    # dic={ent1:[triple1,triple2,triple3],ent2:[]...}
    ent2triple_dict = _build_dict_by_first_element(attribute_triples)
    # attr node id starts from num_ent
    triple2id_dict = _build_dict_by_list(attribute_triples, num_ent)

    def get_time_value(x):
        return time_id2value[x[1]]

    # dynamic graph
    edges_list = []
    labels_list = []

    for window in [1, 2, 3]:
        edges = []
        labels = []
        for ent, triples in ent2triple_dict.items():
            grouped_triples = split_by_clusters(triples, clusters)
            for grouped in grouped_triples:
                grouped.sort(key=get_time_value, reverse=False)
                for i in range(len(grouped)):
                    triple_i = grouped[i]
                    node_i = triple2id_dict[triple_i]
                    for j in range(i + 1, min(len(grouped), window)):
                        triple_j = grouped[j]
                        node_j = triple2id_dict[triple_j]
                        val_val_edge_i_j = [node_i, node_j]
                        val_val_label_i_j = triple_i[2]
                        # val_val_label_i_j = [triple_i[2], [triple_j][2]]
                        edges.append(val_val_edge_i_j)
                        labels.append(val_val_label_i_j)

        # avoid None
        if len(edges) == 0 or len(labels) == 0:
            edges = [[num_ent, num_ent + 1]]
            labels = [[0, 1]]

        edges_list.append(edges)
        labels_list.append(labels)

    return edges_list, labels_list
