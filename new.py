import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import opt_einsum as oe
import wandb

wandb.init(project="toroid")

# from apex import amp
# print(torch.cuda.get_device_name(0))
# torch.manual_seed(0)
# device = torch.device("cuda")
device = torch.device("cpu")


class MyException(Exception):
    pass


def two_to_one_coordinate(size, cell):
    return cell[0] * size[0] + cell[1]


def one_to_two_coordinate(size, coord):
    i = coord // size[0]
    j = coord - (i * size[0])
    return torch.tensor([i, j])


def toroid_distance2d(size, cellA, cellB):
    wrap_dist = torch.abs(size + torch.min(cellA, cellB) - torch.max(cellA, cellB))
    no_wrap_dist = torch.abs(cellA - cellB)
    min_dist = torch.min(wrap_dist, no_wrap_dist) ** 2
    return torch.sum(min_dist)


def toroid_distance1d(size, coordA, coordB):
    return toroid_distance2d(
        size, *[one_to_two_coordinate(size, i) for i in [coordA, coordB]]
    )


def calculate_distance_grid(size):
    return torch.tensor(
        [
            [
                toroid_distance1d(size, torch.tensor([i]), torch.tensor([j]))
                for i in range(torch.prod(size))
            ]
            for j in range(torch.prod(size))
        ],
        dtype=torch.float,
    )

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Grid(nn.Module):
    def __init__(self, size):
        # Size is a torch tensor, BEFORE HyperGrid
        super(Grid, self).__init__()
        # import pdb; pdb.set_trace()
        self.new_size = (size ** 2).tolist()

        # self.scaling_threshold = scaling_threshold
        self.grid = torch.rand(self.new_size, device=device)*2/(self.new_size[0])
        # self.grid = torch.ones(self.new_size, device=device)/(self.new_size[0])
        self.grid.requires_grad=True

        lr = 1e-7

        wandb.config.lr = lr

        self.grid_optim= optim.SGD([self.grid], lr=lr)
        self.grid_scheduler=optim.lr_scheduler.ReduceLROnPlateau(self.grid_optim,patience=1000)
        # for 10x10 3e-3

        self.distance_grid = calculate_distance_grid(size).to(device)
        eq = "ac,bd,cd,ab->"

        ops = (self.new_size, self.new_size, self.distance_grid, self.distance_grid)
        constants = [2, 3]

        self.expr = oe.contract_expression(
            eq, *ops, constants=constants, optimize="optimal"
        )

    def grid_loss(self, grid):
        return self.expr(grid, grid, backend="torch")

    def forward(self,ratio):
        self.grid_optim.zero_grad()
        gridloss = self.grid_loss(self.grid)
        # scale_loss = self.scaling_loss()
        self.ras()
        ras_loss=F.mse_loss(self.grid,self.ras_grid,reduction="sum")
        # inv_loss=self.inv_maximum_deviation()/100
        # varianceloss=self.variance_loss()
        # total_loss = gridloss * (1 - ratio) + scale_loss * ratio
        # total_loss = gridloss * (1 - ratio) + varianceloss* ratio
        # total_loss = gridloss * (1 - ratio) + scale_loss* ratio*100
        total_loss = gridloss * (1 - ratio) + ras_loss* ratio
        total_loss.backward(retain_graph=True)
        self.grid_optim.step()
        self.grid.data.abs_()
        # print("Real Loss:", self.expr(self.grid, self.grid, backend="torch"))
        print("Grid Loss:", gridloss)
        print("Ras Loss:", ras_loss)
        self.discretization()
        real_loss = self.grid_loss(self.discrete)
        print("Real Discrete Loss:",real_loss)
        # self.grid_scheduler.step(real_loss)
        self.grid_scheduler.step(total_loss)
        # print("Inv Max Loss:",inv_loss)
        # wandb.log(
                # {"ratio":ratio,"grid_loss": gridloss, "scale_loss": scale_loss, "real_loss": real_loss}
        # )
        wandb.log(
                {"lr":get_lr(self.grid_optim),"grid_loss": gridloss, "ras_loss": ras_loss, "real_loss": real_loss}
        )
        return total_loss

    def scaling_loss(self):
        row_sum = torch.sum(self.grid, axis=0)
        row_column = torch.sum(self.grid, axis=1)

        # This is not mean loss
        loss = torch.sum((row_sum - 1) ** 2 + (row_column - 1) ** 2)
        return loss*self.new_size[0]

    def ras(self):
        # Where proportion is % of original
        scale_a= torch.diag(1 / torch.sum(self.grid, axis=1))
        grid = scale_a@ self.grid
        scale_b= torch.diag(1 / torch.sum(grid, axis=0))
        grid = grid @ scale_b

        self.ras_grid=grid
        return grid

    def discretization(self):
        grid = torch.abs(self.grid)
        discrete_grid = torch.zeros(self.new_size, device=device)
        while (grid >= 0).any():
            index = divmod(torch.argmax(grid).item(), self.new_size[0])
            discrete_grid[index] = 1
            grid[index[0]] = -1
            grid[:, index[1]] = -1
        self.discrete = discrete_grid

    def variance_loss(self):
        row_var=torch.std(self.grid,dim=0)
        col_var=torch.std(self.grid,dim=1)
        return torch.sum(1/row_var+1/col_var)

    def inv_maximum_deviation(self):
        max_row=torch.max(self.grid,dim=0,keepdim=True)[1]
        max_col=torch.max(self.grid,dim=1,keepdim=True)[1]
        squared_row=torch.sqrt((self.grid-max_row)**2)
        squared_col=torch.sqrt((self.grid-max_col)**2)
        return -torch.sum(squared_row+squared_col)




if __name__ == "__main__":
    size = torch.tensor([10, 10])
    wandb.config.size = size[0].item()
    grid = Grid(size)
