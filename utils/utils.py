from prettytable import PrettyTable


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


