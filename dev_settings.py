class DevSettings:
    def __init__(self):
        self.context_size = 10
        self.num_gcn_layers = 2

        # ablation study
        self.cluster_method = 'both'  # ['both','occ','semantic','random','no']
        self.encoder_method = 'both'  # ['both','no_attr','no_ent','undirected','no_window']

        # training
        self.train_on_batch = False
        self.batch_size = 2048
        self.print_loss = True
        self.print_loss_freq = 50
        self.grid_search = False
        self.is_dev = False
        self.max_epoch = 100
        self.lr = 7e-3
        self.l2_regularization = 0.0001
        self.test_on_all_channels = False

        # cluster
        self.use_cluster = True
        self.eps = 0.1
        self.min_sample = 5

        self.use_time_graph = False

        # models
        self.use_time_sequence = True
        # h_e
        self.h_e_method = 'GCN'  # [None, GCN, GAT]
        # h
        self.h_method = 'add'  # [None, add, cat, weight]
        self.h_norm = False
        self.residual = False
        # h_c
        self.use_dense_graph = False
        self.context_method = 'AttrGraph'  # [AttSeq, GRU, GCN, GAT, CNN, DenseGraph, AttrGraph]
        self.attr_edge_direction = 'ordered'  # [ordered; bi-direction]
        self.edge_direction = 'j->e'  # [e->j; j->e; bi-direction];

        self.attr_encoder = 'GCN'  # [GCN, GAT, DEV]
        self.attr_num_layer = 2
        if self.context_method == 'AttrGraph':
            self.attr_num_layer = 1

        self.ent_attr_encoder = 'GAT'  # [GCN, GAT, DEV]
        self.ent_num_layer = 1

        self.use_edge_list = True
        self.list_size = 3
        self.ent_at_every_layer = False
        # path
        self.use_auxiliary_path = False
