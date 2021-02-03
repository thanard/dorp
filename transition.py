import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Transition(object):

    def __init__(self, enable_factored, enable_full, z_dim, num_onehots):
        self.enable_factored = enable_factored
        self.enable_full = enable_full
        self.num_onehots = num_onehots
        if enable_full:
            full_graph_dim = [z_dim for _ in range(num_onehots)] * 2
            self.full_graph = torch.zeros(*full_graph_dim,
                                          dtype=torch.bool,
                                          device=torch.device("cuda:0"))
        if enable_factored:
            factored_graph_dim = [num_onehots, z_dim, z_dim]
            self.factored_graph = torch.zeros(*factored_graph_dim,
                                              dtype=torch.bool,
                                              device=torch.device("cuda:0"))

    def add(self, node_cur, node_next):
        # node: batch_size x num_onehots (valued between 0 to z_dim-1)
        if self.enable_factored:
            for onehot_idx in range(self.num_onehots):
                self.factored_graph[onehot_idx,
                                    node_cur[:, onehot_idx],
                                    node_next[:, onehot_idx]] = 1
        if self.enable_full:
            self.full_graph[torch.cat([node_cur.t(), node_next.t()])] = 1

    def remove(self, z_cur, z_next):
        # z: batch_size x z_dim
        if self.enable_factored:
            for onehot_idx in range(self.num_onehots):
                self.factored_graph[onehot_idx,
                                    z_cur[:, onehot_idx],
                                    z_next[:, onehot_idx]] = 0
        if self.enable_full:
            self.full_graph[torch.cat([z_cur.t(), z_next.t()])] = 0

    def reset(self):
        if self.enable_factored:
            self.factored_graph[:] = 0
        if self.enable_full:
            self.full_graph[:] = 0

    def find_path(self, mode):
        if mode == 'factor':
            assert self.enable_factored
            np_graph = self.factored_graph.cpu().numpy()
            np_graph = np_graph.reshape()
            costs, predecessors = shortest_path(np_graph,
                                                unweighted=True,
                                                directed=True,
                                                return_predecessors=True)

# def get_factored_edges(model, dataloader):
#     edges = torch.zeros(model.num_onehots, model.z_dim, model.z_dim,
#                         device=torch.device('cuda:0'))
#     for idx, (anchor, pos, act) in enumerate(dataloader):
#         o = anchor.cuda(non_blocking=True)
#         o_next = pos.cuda(non_blocking=True)
#         z_cur = model.encode(o, vis=True)
#         z_next = model.encode(o_next, vis=True)
#         for onehot_idx in range(model.num_onehots):
#             edges[onehot_idx, z_cur[:, onehot_idx], z_next[:, onehot_idx]] += 1
#     return edges