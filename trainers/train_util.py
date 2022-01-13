import torch

def copy_gradient(global_model, local_model):
    for global_param, local_param in zip(filter(lambda p: p.requires_grad, global_model.parameters()),
                                       filter(lambda p: p.requires_grad, local_model.parameters())):
            #if local_param.grad is None:
                #global_param._grad = torch.zeros(global_param.shape)
            #else:
            global_param._grad = local_param.grad.clone().cpu()

def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
        player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()

def transfer_gradient_to_shared(gradient, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape)
            elif gpu_id < 0:
                param._grad = gradient[i]
            else:
                param._grad = gradient[i].cpu()

        i += 1

def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            # theta[name] = torch.tensor(
            #     param_copied,
            #     requires_grad=True,
            #     device=torch.device("cuda:{}".format(gpu_id)),
            # )
            # Changed for pythorch 0.4.1.
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta

def SGD_step(theta, grad, lr):
    theta_i = {}
    j = 0
    for name, param in theta.items():
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param
        j += 1

    return theta_i