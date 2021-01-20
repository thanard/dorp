import torch.optim as optim
from utils.gen_utils import *
from utils.dataset import *
from utils.visualize import *
from model import *
from planning import *
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.dataset import TrajectoryDataset
import time

# Train representation
def train(env,
          actor,
          num_epochs,
          output_dir,
          step_size=1,
          vis_freq=10,
          plan_freq=50,
          writer=None,
          loss_form='ce',
          lr=1e-3,
          ce_temp=1.0,
          baseline='',
          n_traj=50,
          len_traj=1000,
          dataset=None, # None if using online data collection
          kwargs=None):
    # Setting up model
    torch.backends.cudnn.benchmark = True
    model = actor.cpc_model
    model.cuda()
    C_solver = optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())

    # Setting up data
    data = dataset if type(dataset) is np.ndarray \
        else get_sample_transitions(env, n_traj, len_traj,
                                    switching_factor_freq=kwargs['switching_factor_freq'])
    dataloader = torch.utils.data.DataLoader(
        TrajectoryDataset(data, step_size, kwargs['random_step_size']),
        batch_size=model.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=0,
        pin_memory=True
    )
    data_size = len(data) * len(data[0])
    n_batches = data_size // model.batch_size
    if not os.path.exists(kwargs['dataset_path']):
        os.makedirs(kwargs['dataset_path'])
        np.save(os.path.join(kwargs['dataset_path'], 'dataset.npy'),
                data)

    iter_idx = 0
    start_start = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        print("-----Epoch %d------" % epoch)
        # start = time.time()
        for anchors, positives in dataloader:
            # print("preprocessing", round(1e3*(time.time() - start)))
            # start = time.time()
            o = anchors.cuda(non_blocking=True)
            # print("np_2_var_1", round(1e3*(time.time() - start)))
            # start = time.time()
            o_next = positives.cuda(non_blocking=True)
            # print("np_2_var_2", round(1e3*(time.time() - start)))
            # start = time.time()
            ### Compute loss
            z_a, attn_maps_o = model.encode(o, continuous=True)
            z_pos, attn_maps_o_next = model.encode(o_next)
            # print("compute embeddings", round(1e3*(time.time() - start)))
            # start = time.time()
            C_loss = model.get_loss(loss_form, z_a, z_pos, ce_temp=ce_temp)
            # print("compute loss", round(1e3*(time.time() - start)))
            # start = time.time()

            C_loss.backward()
            C_solver.step()
            reset_grad(params)
            # print("backward pass", round(1e3*(time.time() - start)))
            # start = time.time()

            if iter_idx % 100 == 0:
                print("fps", iter_idx*model.batch_size/(time.time() - start_start))
                print("Total time", round(time.time()-start_start))
                log_loss = C_loss.item()
                print("## Iter %d/%d ##" % (iter_idx, n_batches))
                print("C_loss: ", log_loss)
                print()
            iter_idx += 1

        model_fname = os.path.join(output_dir, 'cpc-model')

        # ### Log visualizations to tensorboard
        if writer:
            log(env,
                writer,
                log_loss,
                o,
                o_next,
                attn_maps_o,
                attn_maps_o_next,
                model,
                epoch,
                vis_freq,
                model_fname,
                data)

            log_planning_evals(writer,
                               actor,
                               env,
                               epoch,
                               plan_freq,
                               n_traj=n_traj,
                               len_traj=len_traj)

def log(env,
        writer,
        avg_loss,
        o,
        o_next,
        attn_maps_o,
        attn_maps_o_next,
        model,
        epoch,
        vis_freq,
        model_fname,
        buffer):
    model.eval()

    writer.add_scalar('Train/training loss', avg_loss, epoch)
    writer.add_scalar('Train/k', model.k, epoch)

    if epoch % vis_freq == 0:
        hammings = get_hamming_dists_samples(model, buffer)
        for n in range(model.num_onehots+1):
            writer.add_scalar('Eval_hamming/avg_hamming_%d' % n, np.sum(hammings == n), epoch)
        cluster_fig = visualize_representations(env, model)

        if env.name == 'gridworld':
            if model.num_onehots == 1:
                writer.add_figure("Eval/clusters_single_agent",
                                  cluster_fig,
                                  global_step=epoch)
            elif model.num_onehots == 2:
                plot_0, onehot_0_fig_0, onehot_1_fig_0, plot_1, onehot_0_fig_1, onehot_1_fig_1 = cluster_fig
                writer.add_figure("clusters_vis_double_agent",
                                  [plot_0, plot_1],
                                  global_step=epoch)
                writer.add_figure("onehot_0_clusters_vis_double_agent",
                                  [onehot_0_fig_0, onehot_0_fig_1],
                                  global_step=epoch)
                writer.add_figure("onehot_1_clusters_vis_double_agent",
                                  [onehot_1_fig_0, onehot_1_fig_1],
                                  global_step=epoch)

        elif env.name.startswith('key') and cluster_fig:
            fig_key, fig_no_key = cluster_fig
            writer.add_figure("Clustering with and without key on grid",
                              [fig_key, fig_no_key],
                              global_step=epoch)

        if model.encoder_form.startswith('cswm'):
            # Log Training Data
            training_imgs = [o[0],
                            o_next[0]]
            writer.add_image("training_samples",
                             make_grid(torch.clamp(
                                 torch.stack(training_imgs, dim=0), 0, 1), nrow=2, pad_value=0.5),
                             global_step=epoch)

            # Log Activation Map
            act_imgs = []
            for onehot_idx in range(model.num_onehots):
                act_imgs.append(attn_maps_o[0, onehot_idx, None])
                act_imgs.append(attn_maps_o_next[0, onehot_idx, None])
            writer.add_image("activation_maps",
                             make_grid(torch.clamp(
                                 torch.stack(act_imgs, dim=0), 0, 1), nrow=2, pad_value=0.5),
                             global_step=epoch)

        # Visualize onehot Distribution
        factorization_hists = get_factorization_hist(env, model)
        if factorization_hists:
            hammings_hist, onehots_hist = factorization_hists
            writer.add_figure('hamming distance distribution fixing each agent',
                              hammings_hist,
                              global_step=epoch)
            writer.add_figure('which onehot changes fixing each independent entity',
                              onehots_hist,
                              global_step=epoch)
    if epoch > 10:
        torch.save(model.state_dict(), model_fname)

    plt.close('all')

def log_planning_evals(writer,
                       actor,
                       env,
                       epoch,
                       plan_freq,
                       n_traj,
                       len_traj):

    ### Planning
    actor.cpc_model.eval()
    if writer and epoch % plan_freq == 0:
        # currently uses oracle dynamics for evaluation
        if env.name == 'gridworld':
            factorized_planning_success_rate = get_planning_success_rate(actor,
                                                                         env,
                                                                         10,
                                                                         factorized=True,
                                                                         oracle=True,
                                                                         n_traj=n_traj,
                                                                         len_traj=len_traj,
                                                                         )
            writer.add_scalar('Eval/execute_plan_success_rate_factorized_graphs_with_replan',
                              factorized_planning_success_rate,
                              epoch)
        elif env.name.startswith('key'):
            factorized_planning_success_rate = get_planning_success_rate(actor,
                                                                         env,
                                                                         10,
                                                                         factorized=False,
                                                                         oracle=True,
                                                                         n_traj=n_traj,
                                                                         len_traj=len_traj,
                                                                         )
            writer.add_scalar('Eval/execute_plan_success_rate_full_graph_with_replan',
                              factorized_planning_success_rate,
                              epoch)

