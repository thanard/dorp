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
          dataset=None,
          datapath=None,
          graph=None,
          kwargs=None):
    # Setting up model
    torch.backends.cudnn.benchmark = True
    model = actor.cpc_model
    model.cuda()
    C_solver = optim.Adam(model.parameters(), lr=lr)
    # params = list(model.parameters())

    # Setting up data
    dataloader = {}
    data_config = {
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
    }
    valid_data_size = int(1e5)
    data_n_traj = {
        "train": n_traj,
        "valid": valid_data_size // len_traj
    }

    for key, value in data_n_traj.items():
        if type(dataset[key]['obs']) != np.ndarray or type(dataset[key]['act']) != np.ndarray:
            dataset[key]['obs'], dataset[key]['act'] = get_sample_transitions(
                env,
                value,
                len_traj,
                switching_factor_freq=kwargs['switching_factor_freq'])
            for key2 in ['obs', 'act']:
                datapath[key][key2].parent.mkdir(parents=True, exist_ok=True)
                np.save(datapath[key][key2], dataset[key][key2])
        if key == "train":
            dataloader[key] = torch.utils.data.DataLoader(
                TrajectoryDataset(dataset[key], step_size, kwargs['random_step_size']),
                batch_size=model.batch_size,
                **data_config
            )
        elif key == "valid":
            dataloader[key] = torch.utils.data.DataLoader(
                TrajectoryDataset(dataset[key], 1),
                batch_size=4096,
                pin_memory=True
            )

    data_size = len(dataset['train']['obs']) * len(dataset['train']['obs'][0])
    n_batches = data_size // model.batch_size

    iter_idx = 0
    start_start = time.time()
    for epoch in range(num_epochs + 1):
        model.train()
        graph.reset()
        print("-----Epoch %d------" % epoch)
        # start = time.time()
        for anchors, positives, actions in dataloader['train']:
            # print("preprocessing", round(1e3*(time.time() - start)))
            # start = time.time()
            o = anchors.cuda(non_blocking=True)
            # print("np_2_var_1", round(1e3*(time.time() - start)))
            # start = time.time()
            o_next = positives.cuda(non_blocking=True)
            # print("np_2_var_2", round(1e3*(time.time() - start)))
            # start = time.time()
            ### Compute loss
            z_cur, node_cur, attn_maps_o = model.encode(o, continuous=True)
            z_next, node_next, attn_maps_o_next = model.encode(o_next)
            # print("compute embeddings", round(1e3*(time.time() - start)))
            # start = time.time()
            C_loss, loss_info = model.get_loss(loss_form, z_cur, z_next, ce_temp=ce_temp)
            # print("compute loss", round(1e3*(time.time() - start)))
            # start = time.time()

            # Not perfect as the representation is constantly changing.
            import pdb
            pdb.set_trace()
            graph.add(node_cur, node_next)

            if iter_idx % 100 == 0:
                fps = iter_idx*model.batch_size/(time.time() - start_start)
                total_time = round(time.time()-start_start)
                log_loss = C_loss.item()
                print("## Iter %d/%d ##" % (iter_idx, n_batches))
                print("C_loss: ", log_loss)
                print("fps", fps)
                print("Total time", total_time)
                print()
            iter_idx += 1

            # Checking visualization
            if epoch == 0:
                break

            # Update parameters
            C_loss.backward()
            C_solver.step()
            C_solver.zero_grad()

        model_fname = output_dir / 'cpc-model'

        ### Log visualizations to tensorboard
        if writer:
            log(env,
                writer,
                log_loss,
                fps,
                total_time,
                loss_info,
                o,
                o_next,
                attn_maps_o,
                attn_maps_o_next,
                model,
                epoch,
                vis_freq,
                model_fname,
                dataloader['valid'])

            log_planning_evals(writer,
                               actor,
                               env,
                               graph,
                               epoch,
                               plan_freq)

def log(env,
        writer,
        log_loss,
        fps,
        total_time,
        loss_info,
        o,
        o_next,
        attn_maps_o,
        attn_maps_o_next,
        model,
        epoch,
        vis_freq,
        model_fname,
        valid_dataloader):
    model.eval()

    # Log every epoch (cheap) -- Train constants and Time
    writer.add_scalar('Train/training loss', log_loss, epoch)
    writer.add_scalar('Train/k', model.k, epoch)
    if loss_info:
        for k, v in loss_info.items():
            writer.add_scalar('Train/%s' % k, v, epoch)

    writer.add_scalar('Time/fps', fps, epoch)
    writer.add_scalar('Time/total_time', total_time, epoch)

    # Log every vis freq (expensive) -- Hamming Distances
    if epoch % vis_freq == 0:
        hammings, act_labels, entity_idx_to_onehot, entity_idx_to_hamming = \
            get_hamming_dists_samples(model, valid_dataloader)
        for n in range(model.num_onehots+1):
            writer.add_scalar('Eval/avg_hamming_%d' % n,
                              (hammings == n).sum().item(), epoch)
            if env.name != 'gridworld':
                continue
            for j in range(model.num_onehots):
                writer.add_scalar('Eval/entity_to_hamming/idx_%d_to_dist_%d' % (j, n),
                                  entity_idx_to_hamming[j, n], epoch)
                if n == model.num_onehots:
                    continue
                writer.add_scalar('Eval/entity_to_onehot/idx_%d_to_onehot_%d' % (n, j),
                                  entity_idx_to_onehot[n, j], epoch)
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
            # act_imgs = []
            # for onehot_idx in range(model.num_onehots):
            #     act_imgs.append(attn_maps_o[0, onehot_idx, None])
            #     act_imgs.append(attn_maps_o_next[0, onehot_idx, None])
            # writer.add_image("activation_maps",
            #                  make_grid(torch.clamp(
            #                      torch.stack(act_imgs, dim=0), 0, 1), nrow=2, pad_value=0.5),
            #                  global_step=epoch)

    # Visualize onehot Distribution
    if epoch % 200 == 0 and epoch != 0:
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
                       graph,
                       epoch,
                       plan_freq):

    ### Planning
    actor.cpc_model.eval()
    if writer and epoch % plan_freq == 0:
        # currently uses oracle dynamics for evaluation
        success_rates = get_planning_success_rate(actor,
                                                  env,
                                                  graph,
                                                  n_trials=10,
                                                  oracle=True)
        for k, sr in success_rates.items():
            writer.add_scalar('Eval/success_rate_%s_replan_oracle' % k,
                              sr,
                              epoch)
        # elif env.name.startswith('key'):
        #     factorized_planning_success_rate = get_planning_success_rate(actor,
        #                                                                  env,
        #                                                                  n_trials=10,
        #                                                                  oracle=True)
        #     writer.add_scalar('Eval/execute_plan_success_rate_full_graph_with_replan',
        #                       factorized_planning_success_rate,
        #                       epoch)

