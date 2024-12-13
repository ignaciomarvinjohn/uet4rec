import torch
import torch.nn.functional as F
import torch.distributions as dist

# Double Q-Learning loss function
def dq_loss_fn(forward_output, action, reward, discount):
    # select the best action based on main Q-values
    with torch.no_grad():
        best_action = torch.argmax(forward_output['main_q_values'], dim=1)
        double_q_bootstrapped = forward_output['target_q_values'][torch.arange(forward_output['target_q_values'].size(0)), best_action]
        target = reward + discount * double_q_bootstrapped

    # get the Q-value for the current state-action pair
    qa_state = forward_output['current_main_out']['q_values'][torch.arange(forward_output['current_main_out']['q_values'].size(0)), action]

    # compute TD error and loss
    td_error = target - qa_state
    loss = 0.5 * torch.square(td_error).mean()

    return loss, qa_state


# AWAC loss function
def awac_loss_fn(forward_output, qa_state, action, discount, negative_reward):
    # get the greedy action (argmax Q-values under current policy)
    greedy_action = torch.argmax(forward_output['current_main_out']['q_values'], dim=1)

    # get the baseline value V(s) for greedy actions
    _, baseline_value = dq_loss_fn(forward_output, greedy_action, negative_reward, discount)

    # calculate advantage A(s, a) = Q(s, a) - V(s)
    advantage = qa_state - baseline_value

    # compute advantage-based weights
    weights = torch.clamp_max(torch.exp(advantage), 100.0)

    # policy distribution and log probabilities
    pi_distribution = dist.Categorical(logits=forward_output['current_main_out']['q_values'])
    logp_pi = pi_distribution.log_prob(action)

    # weighted policy loss
    loss = (-logp_pi * weights).mean()

    return loss



# InfoNCE loss function
def infoNCE_loss_fn(current_state, augmented_state):
    # concatenate the states
    batch_logits = torch.cat([current_state, augmented_state])
    
    # compute for cosine similarity
    cos_sim = F.cosine_similarity(batch_logits[:, None, :], batch_logits[None, :, :], dim=-1)
    
    # mask the diagonal (self-similarity) to avoid the states from comparing to themselves
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # identify positive pairs
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    
    # add temperature
    temperature = 0.07
    cos_sim = cos_sim / temperature
    
    # compute for the InfoNCE loss
    loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    loss = loss.mean()
    
    return loss

