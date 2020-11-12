import torch.optim as optim
from utils.gen_utils import *
from utils.dataset import *
from utils.visualize import *
from model import *
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time

# Train representation
def train(env,
          actor,
          num_epochs,
          output_dir,
          vis_freq=10,
          plan_freq=50,
          writer=None,
          loss_form='ce',
          lr=1e-3,
          ce_temp=1.0,
          baseline='',
          n_traj=50,
          len_traj=1000):

    losses = []
    est_lowerbounds = []
    model = actor.cpc_model
    model.cuda()
    C_solver = optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        print("-----Epoch %d------" % epoch)
        epoch_losses = []
        epoch_lb = []
        buffer = get_sample_transitions(env, n_traj, len_traj)
        data_size = n_traj * len_traj
        n_batches = data_size // model.batch_size
        for it in range(n_batches):
            anchors, positives = sample_anchors_positives(buffer, model.batch_size)
            o = np_to_var(np.transpose(anchors, (0, 3, 1, 2)))
            o_next = np_to_var(np.transpose(positives, (0, 3, 1, 2)))
            save_image(o, "test.png", padding=5)
            save_image(o_next, "test_1.png", padding=5)

            ### Compute loss
            z_a = model.encode(o, continuous=True)
            z_pos = model.encode(o_next)
            if len(z_a) == 3:
                z_a, attn_reg_a, attn_maps_o = z_a
            if len(z_pos) == 3:
                z_pos, attn_reg_pos, attn_maps_o_next = z_pos

            C_loss = get_loss(loss_form, model, z_a, z_pos, ce_temp=ce_temp)
            epoch_losses.append(C_loss.item())
            lb = np.log(model.batch_size) - C_loss.item()
            epoch_lb.append(lb)

            if epoch == 0:
                break

            C_loss.backward()
            C_solver.step()

            if it % 100 == 0:
                print("C_loss: ", C_loss)
                print("Est lowerbound: ", np.log(model.batch_size) - C_loss.item())
                print()
            reset_grad(params)

        avg_loss = np.mean(np.array(epoch_losses))
        avg_lb = np.mean(np.array(epoch_lb))
        losses.append(avg_loss)
        est_lowerbounds.append(avg_lb)
        save_plots(output_dir, losses, est_lowerbounds)

        model_fname = os.path.join(output_dir, '%d-agents-model' % env.n_agents)
        #
        # ### Logging visualizations to tensorboard
        if writer:
            log(env,
                writer,
                avg_loss,
                o,
                o_next,
                attn_maps_o,
                attn_maps_o_next,
                model,
                epoch,
                vis_freq,
                model_fname)

        ### Planning


        #         cluster_fig = visualize_clusters_online_single(model, n_agents, grid_n)
        #         writer.add_figure("Eval/clusters_single_agent",
        #                           cluster_fig,
        #                           global_step=epoch)
        #     if model.num_onehots == 2:
        #         vis_data = visualize_clusters_online_double(model, n_agents, grid_n)
        #         plot_0, onehot_0_fig_0, onehot_1_fig_0, plot_1, onehot_0_fig_1, onehot_1_fig_1 = vis_data
        #         writer.add_figure("clusters_vis_double_agent",
        #                           [plot_0, plot_1],
        #                           global_step=epoch)
        #         writer.add_figure("onehot_0_clusters_vis_double_agent",
        #                           [onehot_0_fig_0, onehot_0_fig_1],
        #                           global_step=epoch)
        #         writer.add_figure("onehot_1_clusters_vis_double_agent",
        #                           [onehot_1_fig_0, onehot_1_fig_1],
        #                           global_step=epoch)
        #
        #     if model.encoder_form.startswith('cswm'):
        #         # Log Training Data
        #         training_imgs = [model.encoder.to_rgb(o)[0],
        #                     model.encoder.to_rgb(o_next)[0]]
        #         writer.add_image("training_samples",
        #                          make_grid(torch.clamp(
        #                              torch.stack(training_imgs, dim=0), 0, 1), nrow=2, pad_value=0.5),
        #                          global_step=epoch)
        #
        #         # Log Activation Map
        #         act_imgs = []
        #         for onehot_idx in range(model.num_onehots):
        #             act_imgs.append(attn_maps_o[0, onehot_idx, None])
        #             act_imgs.append(attn_maps_o_next[0, onehot_idx, None])
        #         writer.add_image("activation_maps",
        #                          make_grid(torch.clamp(
        #                              torch.stack(act_imgs, dim=0), 0, 1), nrow=2, pad_value=0.5),
        #                          global_step=epoch)
        #
        #     # Visualize Onehot Distribution
        #     hammings_hist, onehots_hist = test_factorization_fix_agents(model, n_agents, grid_n, circular=circular, allow_overlap=allow_overlap)
        #     writer.add_figure('hamming distance distribution fixing each agent',
        #                       hammings_hist,
        #                       global_step=epoch)
        #     writer.add_figure('which onehot changes fixing each agent',
        #                       onehots_hist,
        #                       global_step=epoch)
        # ### Planning
        # if writer and epoch % plan_freq == plan_freq-1:
        #     n_batches = int((n_agents*1e5) // this_batch_size)
        #     timestamp = time.time()
        #     graph, onehot_graphs = create_graph_sample_transitions(model, grid_n, n_agents, n_batches=n_batches, batch_size=this_batch_size)
        #     writer.add_scalar('Time/planning/create_graph_time', time.time() - timestamp, epoch)
        #     timestamp = time.time()
        #     execute_factorized_plan_success_rate = get_factorized_planning_success_rate(model, onehot_graphs, n_agents, grid_n, h_distance, n_plans=40)
        #     writer.add_scalar('Time/planning/fact_planning_time', time.time() - timestamp, epoch)
        #     timestamp = time.time()
        #     execute_factorized_plan_success_rate_with_replan = get_factorized_planning_success_rate_with_replan(model, onehot_graphs, n_agents, grid_n, h_distance, n_plans=40, n_retry=5)
        #     writer.add_scalar('Time/planning/fact_replanning_time', time.time() - timestamp, epoch)
        #     print("success rate factorized graphs", execute_factorized_plan_success_rate)
        #     print("success rate factorized graphs with replan", execute_factorized_plan_success_rate_with_replan)
        #     for n in range(n_agents + 1):
        #         writer.add_scalar('Eval_hamming/avg_hamming_%d' % n, np.sum(hammings == n), epoch)
        #     writer.add_scalar('Eval/execute_plan_success_rate_factorized_graphs', execute_factorized_plan_success_rate, epoch)
        #     writer.add_scalar('Eval/execute_plan_success_rate_factorized_graphs_with_replan', execute_factorized_plan_success_rate_with_replan,
        #                       epoch)
        #
        # if epoch > 50:
        #     torch.save(model.state_dict(), model_fname)
        #
        # plt.close('all')

    # return model, losses, est_lowerbounds

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
        model_fname):

    writer.add_scalar('Train/training loss', avg_loss, epoch)
    writer.add_scalar('Train/k', model.k, epoch)

    if epoch % vis_freq == 0:
        hammings = get_hamming_dists_samples(env, model, n_batches=64, batch_size=256)
        for n in range(env.n_agents + 1):
            writer.add_scalar('Eval_hamming/avg_hamming_%d' % n, np.sum(hammings == n), epoch)
        cluster_fig = visualize_representations(env, model)
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

        # Visualize Onehot Distribution
        hammings_hist, onehots_hist = test_factorization_fix_agents(env, model)
        writer.add_figure('hamming distance distribution fixing each agent',
                          hammings_hist,
                          global_step=epoch)
        writer.add_figure('which onehot changes fixing each agent',
                          onehots_hist,
                          global_step=epoch)
    ### Planning
    # if writer and epoch % plan_freq == plan_freq-1:
    #     n_batches = int((n_agents*1e5) // this_batch_size)
    #     timestamp = time.time()
    #     graph, onehot_graphs = create_graph_sample_transitions(model, grid_n, n_agents, n_batches=n_batches, batch_size=this_batch_size)
    #     writer.add_scalar('Time/planning/create_graph_time', time.time() - timestamp, epoch)
    #     timestamp = time.time()
    #     execute_factorized_plan_success_rate = get_factorized_planning_success_rate(model, onehot_graphs, n_agents, grid_n, h_distance, n_plans=40)
    #     writer.add_scalar('Time/planning/fact_planning_time', time.time() - timestamp, epoch)
    #     timestamp = time.time()
    #     execute_factorized_plan_success_rate_with_replan = get_factorized_planning_success_rate_with_replan(model, onehot_graphs, n_agents, grid_n, h_distance, n_plans=40, n_retry=5)
    #     writer.add_scalar('Time/planning/fact_replanning_time', time.time() - timestamp, epoch)
    #     print("success rate factorized graphs", execute_factorized_plan_success_rate)
    #     print("success rate factorized graphs with replan", execute_factorized_plan_success_rate_with_replan)
    #     for n in range(n_agents + 1):
    #         writer.add_scalar('Eval_hamming/avg_hamming_%d' % n, np.sum(hammings == n), epoch)
    #     writer.add_scalar('Eval/execute_plan_success_rate_factorized_graphs', execute_factorized_plan_success_rate, epoch)
    #     writer.add_scalar('Eval/execute_plan_success_rate_factorized_graphs_with_replan', execute_factorized_plan_success_rate_with_replan,
    #                       epoch)

    if epoch > 50:
        torch.save(model.state_dict(), model_fname)

    plt.close('all')
