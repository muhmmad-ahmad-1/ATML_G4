import torch


def get_mdl_params(model):
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)



def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)
    
def dict_to_vector(diction :dict):
    # convert a named param/grad dict to a vector
    vec = []
    for _,param in diction.items():
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def set_client_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)



def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape


def vector_to_dict(zeroed_dict, vector):
    index = 0
    for key, param in zeroed_dict.items():
        num_elements = param.numel()
        updated_values = vector[index:index + num_elements]
        zeroed_dict[key] = updated_values.reshape(param.shape)
        index += num_elements
    return zeroed_dict

