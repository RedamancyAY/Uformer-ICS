
import torch
import torch.nn as nn


# 改进于：https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

def Optimizers_with_selective_weight_decay(
    model, lr, weight_decay, optimizer="AdamW", debug=False
):
    """

    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    def _log(info):
        if debug:
            print(info)

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.Conv1d,
    )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    used_fpn = []
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if fpn in used_fpn:
                continue
            used_fpn.append(fpn)
            if fpn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
                _log(fpn + "    Condition 1, add to no_decay")
            elif fpn.endswith("weight"):
                if isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                    _log(fpn + "    Condition 2, add to decay")
                elif isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    _log(fpn + "    Condition 3, add to no_decay")
                else:
                    no_decay.add(fpn)
                    _log(fpn + "    Condition 4, add to decay")
            else:
                no_decay.add(fpn)
                _log(fpn + "    Condition 5, add to no_decay")

    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add('pos_emb')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters {} made it into both decay/no_decay sets!".format(
        str(inter_params),
    )
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters {} were not separated into either decay/no_decay set!".format(
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    if isinstance(optimizer, str):
        assert optimizer in ["AdamW", "Adam", "Lion", "SGD"]
        if optimizer == "AdamW":
            final_optimizer = torch.optim.AdamW(optim_groups, lr=lr)
        elif optimizer == "Adam":
            final_optimizer = torch.optim.Adam(optim_groups, lr=lr)
        elif optimizer.lower() == "sgd":
            final_optimizer = torch.optim.SGD(optim_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(
                "Error, please input right optimizer Name, your input is ", optimizer
            )
    else:
        assert optimizer in [torch.optim.AdamW, torch.optim.Adam]
        final_optimizer = optimizer(optim_groups, lr=lr)
    return final_optimizer
