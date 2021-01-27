import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Transition:
    def __int__(self, enable_factored, enable_full, model):
        self.enable_factored = enable_factored
        self.enable_full = enable_full
        if enable_full:
            full_graph_dim = [model.z_dim for _ in range(model.num_onehots)] * 2
            self.full_graph = torch.zeros(*full_graph_dim, device=torch.device("cuda:0"))
        if enable_factored:
            factored_graph_dim = [model.num_onehots, model.z_dim, model.z_dim]
            self.factored_graph = torch.zeros(*model.num_onehots, device=torch.device("cuda:0"))


    def find_path(self, mode):
        if mode == 'factor':
            assert self.enable_factored
            costs, predecessors = shortest_path(self.enable_factored,
                                                 unweighted=True,
                                                 directed=True,
                                                 return_predecessors=True)
