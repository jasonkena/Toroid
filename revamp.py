# Numerical computation libraries
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import opt_einsum as oe

torch.autograd.set_detect_anomaly(True)

# Misc utilities
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import string
import re


# use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Distance Grid Calculation
"""


def two_to_one_coordinate(size, cell):
    """
    Convert 2d coordinates into 1d coordinates

    :param size [rows, cols]: Dimensions of toroidal grid
    :param cell [row_index, cols_index]: ij matrix like indices
    :return int: 1d coordinates
    """
    return cell[0] * size[0] + cell[1]


def one_to_two_coordinate(size, coord):
    """
    Convert 1d coordinates into 2d coordinates

    :param size [rows, cols]: Dimensions of toroidal grid
    :param coord int: 1d coordinate
    :return [row_index, cols_index]: ij matrix like indices
    """
    # integer division
    i = coord // size[0]
    # division remainder
    j = coord - (i * size[0])
    return torch.tensor([i, j])


def toroid_distance2d(size, cellA, cellB):
    """
    Squared toroidal distance between 2 2d-coordinates

    :param size [rows, cols]: Dimensions of toroidal grid
    :param cell* [row_index, cols_index]: ij matrix like indices
    :return int: Squared toroidal distance
    """
    # distance wrapping around the grid, from leftmost cell to rightmost cell
    wrap_dist = torch.abs(size + torch.min(cellA, cellB) - torch.max(cellA, cellB))
    # distance across the grid
    no_wrap_dist = torch.abs(cellA - cellB)
    # squared distance
    min_dist = torch.min(wrap_dist, no_wrap_dist) ** 2
    return torch.sum(min_dist)


def toroid_distance1d(size, coordA, coordB):
    """
    Squared toroidal distance between 2 1d-coordinates

    :param size [rows, cols]: Dimensions of toroidal grid
    :param coord* int: 1d coordinate
    :return int: Squared toroidal distance
    """
    # computed by conversion to 2d coordinates
    return toroid_distance2d(
        size, *[one_to_two_coordinate(size, i) for i in [coordA, coordB]]
    )


def calculate_distance_grid(size):
    """
    Comparison Grid: matrix with entries representing distances between any 2 tokens within the Original grid

    :param size [rows, cols]: Dimensions of toroidal grid
    :return [rows^2 x cols^2]-shape matrix: Distance grid
    """
    return torch.tensor(
        [
            [
                # distance calculation
                toroid_distance1d(size, torch.tensor([i]), torch.tensor([j]))
                # iteration over source token
                for i in range(torch.prod(size))
            ]
            # iteration over target token
            for j in range(torch.prod(size))
        ],
        dtype=torch.float,
    )


def get_lower_bounds():
    """
    Utility to obtain dictionary of lower-bounds from text-file

    :return {size: lower_bound} Dict: Lower-bounds dictionary
    """
    result = {}
    with open("lower_bounds.txt", "r") as file:
        for line in file:
            key, value = line.split()
            result[int(key)] = int(value)
    return result


class Grid(nn.Module):
    """
    Class to perform Toroidal Grid optimization via Gradient Descent
    """

    def __init__(self, size):
        """
        Intialize Comparison grid via caching mechanism, set of revealed tokens, and lower bounds

        :param size [rows, cols]: Dimensions of toroidal grid
        """
        super(Grid, self).__init__()
        # dimension of Superposition grid
        self.new_size = (size ** 2).tolist()

        # load cached Comparison grid if possible
        filename = os.path.join("cache", str(size[0].item()) + ".pth")
        if os.path.isfile(filename):
            self.expr = torch.load(filename)
        else:
            self.distance_grid = calculate_distance_grid(size).to(device)

            # loss function (missing 1/2 factor added in Grid.grid_loss)
            eq = "ab,cd,ac,bd->"

            ops = (self.new_size, self.new_size, self.distance_grid, self.distance_grid)
            constants = [2, 3]

            self.expr = oe.contract_expression(
                eq, *ops, constants=constants, optimize="optimal"
            )

            # cache Comparison grid
            torch.save(self.expr, filename)

        # set of revealed tokens
        self.revealed = []
        # self-explanatory
        self.lower_bounds = get_lower_bounds()

    def grid_loss(self, grid):
        """
        Calculate Superposition grid loss

        :param grid [rows^2 x cols^2]-shape matrix: Superposition grid
        :return float: Loss
        """
        # add in factor of half. expr is the equation above
        return 0.5 * self.expr(grid, grid, backend="torch")

    def objective_loss(self, grid, final=False):
        """
        Calculate Objective loss: applying discretization and lower bounds

        :param grid [rows^2 x cols^2]-shape matrix: Superposition grid
        :param final bool: If true, assume that discretization makes no change
        :return float: Loss
        """
        # loss of discretized grid
        discretized = self.discretization(grid)

        if final:
            assert torch.all(discretized == grid)

        loss = self.grid_loss(discretized)
        # subtracting lower bounds. apply square root on dimensions to obtain toroidal grid size
        loss = loss - self.lower_bounds[int(grid.size(0) ** 0.5)]
        return loss

    def discretization(self, grid):
        """
        Discretize continuous Superposition grid into a Toroidal grid

        :param grid [rows^2 x cols^2]-shape matrix: Superposition grid
        :return [rows^2 x cols^2]-shape matrix: Discrete grid
        """
        grid = grid.clone()

        # assume all the weights are non-negative
        assert torch.all(grid >= 0)

        # surrogate grid
        discrete_grid = torch.zeros(self.new_size, device=device)

        # iterate while there exists non-transferred weights
        while (grid >= 0).any():
            # obtain indices of maximum weight
            index = divmod(torch.argmax(grid).item(), self.new_size[0])
            # mark this location within the surrogate grid
            discrete_grid[index] = 1
            # remove all other weights from the same token or location (row and column)
            grid[index[0]] = -1
            grid[:, index[1]] = -1
        return discrete_grid

    def ras(self, tensor, n_iter):
        """
        Sinkhorn-Knopp (or RAS) algorithm, to obtain a doubly-stochastic matrix

        :param tensor [rows x cols] matrix: Matrix to be normalized
        :param n_iter int: Number of RAS iterations
        :return [rows x cols] matrix: Normalized matrix
        """
        # assume positive semi-definite matrix (non-negative)
        assert torch.all(tensor >= 0)

        for _ in range(n_iter):
            # obtain column-scaling matrix
            scale_a = torch.diag(1 / torch.sum(tensor, axis=1))
            # normalize columns
            tensor = scale_a @ tensor
            # obtain row-scaling matrix
            scale_b = torch.diag(1 / torch.sum(tensor, axis=0))
            # normalize rows
            tensor = tensor @ scale_b
        return tensor

    def forward(self, optim, n_ras):
        """
        Perform Toroidal grid optimization via gradient descent

        :param optim int: Number of gradient descent iterations
        :param n_ras int: Number of RAS iterations
        :return float: Final Objective loss
        """

        # iterate to reveal each of the tokens
        for n_iter in tqdm(range(self.new_size[0])):
            # initialize doubly-stochastic grid
            grid = torch.ones(self.new_size, device=device) / (
                self.new_size[0] - len(self.revealed)
            )

            # set revealed tokens
            for reveal in self.revealed:
                grid[reveal[0], :] = 0
                grid[:, reveal[1]] = 0
                grid[reveal] = 1

            # enable gradient calculations
            grid.requires_grad = True

            for n_optim in tqdm(range(optim), leave=False):

                # assume non-negative weights
                assert torch.all(grid >= 0)
                # assume gradients have not been calculated
                assert grid.grad is None or torch.all(grid.grad == 0)

                # calculate loss
                gridloss = self.grid_loss(grid)
                # calculate Jacobian matrix
                gridloss.backward()

                # assume positive weights
                assert torch.all(grid.grad > 0)

                # Tensorboard logging code
                global_step = n_iter * optim + n_optim

                discrete_loss = self.objective_loss(grid)
                writer.add_scalar(
                    "realtime_discrete_loss", discrete_loss, global_step,
                )
                writer.add_scalar("grid_loss", gridloss, global_step)
                writer.add_image("grid", grid, global_step, dataformats="HW")
                writer.add_image(
                    "filtered_grid",
                    filter_tensor(grid, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    "filtered_grid_grad",
                    filter_tensor(grid.grad, self.revealed, only=False),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    "grid_grad", grid.grad, global_step, dataformats="HW",
                )

                # perform gradient descent update
                with torch.no_grad():
                    # ignore entries from revealed rows or columns
                    filtered_grad = filter_tensor(grid.grad, self.revealed, only=False)
                    filtered_grid = filter_tensor(grid, self.revealed, only=False)

                    # ensure Zero Line Sum modified Jacobian: all rows and columns sum to 0
                    filtered_grad = self.fix_grad(filtered_grad, n_ras)

                    # determine learning rate required to make each of the weights 0
                    alpha = (
                        filtered_grid[filtered_grad > 0]
                        / filtered_grad[filtered_grad > 0]
                    )
                    if len(alpha):
                        # set learning rate ot be the arithmetic mean
                        # NOTE: maybe use Median
                        lr = torch.mean(alpha)
                        # assert lr != 0
                    else:
                        lr = 0

                    writer.add_scalar("lr", lr, global_step)

                    # gradient descent update
                    filtered_grid = filtered_grid - lr * filtered_grad
                    if torch.any(filtered_grid < 0):
                        # renormalize to ensure positive semi-definite weights
                        filtered_grid = self.fix_grid(filtered_grid)

                    # update grid weights
                    grid.data = filter_tensor(
                        filtered_grid, self.revealed, only=False, fill=grid.data
                    )
                # clear Jacobian matrix for new calculation
                grid.grad.zero_()

                # if not learning, end optimization early
                if lr == 0:
                    break

            # remove weights within revealed columns or rows
            for reveal in self.revealed:
                grid[reveal[0], :] = 0
                grid[:, reveal[1]] = 0

            # reveal highest weight token
            self.revealed.append(divmod(torch.argmax(grid).item(), self.new_size[0]))

        grid = torch.zeros(self.new_size, device=device)
        # set revealed tokens
        for reveal in self.revealed:
            grid[reveal[0], :] = 0
            grid[:, reveal[1]] = 0
            grid[reveal] = 1

        write_grid(grid, os.path.join("grids", save_name))
        final_loss = self.objective_loss(grid, final=True)
        return final_loss

    def fix_grad(self, grad, n_ras):
        """
        Normalize Jacobian to have Zero Line Sum (ZLS): rows and columns summing to 0

        :param grad [rows x cols] matrix: Filtered Jacobian Matrix to be normalized
        :param n_ras int: Number of RAS iterations
        """
        # assume positive gradients
        assert torch.all(grad > 0)
        # offset to subtract, to ensure Zero Line Sum
        offset = torch.tensor(1, device=device, dtype=torch.float) / grad.size(0)
        # Perform RAS and ensure ZLS
        grad = self.ras(grad, n_ras) - offset
        return grad

    def fix_grid(self, grid):
        """
        Renormalize Superposition grid weights with negative weights to be positive semi-definite, whilst maintaining doubly stochastic nature

        :param grid [rows x cols] matrix: Superposition grid to be normalized. Grid is assumed to be doubly-stochastic.
        """
        # Calculate minimum offset required
        offset = -torch.min(grid)
        assert offset >= 0

        # Apply offset
        grid = grid + offset
        # Divide by row/column sum, to ensure doubly stochastic
        grid = grid / (grid.size(0) * offset + 1)

        # Assume non-negative weights
        assert torch.all(grid >= 0)
        return grid


def filter_tensor(tensor, revealed, only, fill=None):
    """
    Utility to filter rows and columns from tensor, or set values within filtered rows and columns

    :param tensor: Matrix to be filtered
    :param revealed [[row_1, col_1],...]: List of indices of revealed tokens
    :param only bool: If True, only include revealed rows and columns; If False, only include unrevealed rows and columns
    :param fill tensor or None: If None, return filtered values; If given, fill is tensor whose values are to be replaced
    """
    # If revealed is empty, set it to filter imaginary indices
    if not len(revealed):
        revealed = [[-1, -1]]

    revealed = torch.tensor(revealed, device=device)
    revealed_rows = revealed[:, 0]
    revealed_cols = revealed[:, 1]

    # Index using booleans
    # if revealed_rows = [1,3,5], size of tensor is 10
    # index_rows = [False, True, False, True, False, True, False, False, False, False]
    index_rows = torch.any(
        revealed_rows.unsqueeze(1)
        == torch.arange(
            # Infer shape of tensor
            (fill.size(0) if fill is not None else tensor.size(0)),
            device=device,
        ).unsqueeze(0),
        dim=0,
    ).unsqueeze(1)
    index_cols = torch.any(
        revealed_cols.unsqueeze(1)
        == torch.arange(
            (fill.size(1) if fill is not None else tensor.size(1)), device=device
        ).unsqueeze(0),
        dim=0,
    ).unsqueeze(0)

    # Negate boolean indices
    if not only:
        index_rows = ~index_rows
        index_cols = ~index_cols

    if fill is None:
        # Index values
        result = tensor[index_rows & index_cols]
        # reshape into square matrix
        size = int(len(result) ** 0.5)
        result = result.view(size, size)
        return result
    else:
        # fill in "fill" values with "tensor" values
        fill[index_rows & index_cols] = torch.flatten(tensor)
        return fill


def read_grid(filename):
    """
    Generate Superposition grid from file

    :param filename str: File to read from
    """
    characters = string.ascii_uppercase + "1234"

    # remove data formatting delimiters
    with open(filename, "r") as file:
        content = re.sub("[\s(),]", "", file.read())

    size = int((len(content) / 2) ** 0.5)
    assert size ** 2 * 2 == len(content)

    tokens = []
    for i in range(0, len(content), 2):
        # obtain 1d coordinates
        tokens.append(
            characters.index(content[i]) * size + characters.index(content[i + 1])
        )
    tokens = torch.tensor(tokens, device=device)
    grid = torch.arange(size ** 2, device=device).unsqueeze(0) == tokens.unsqueeze(1)

    return grid.double()


def write_grid(grid, filename):
    """
    Write grid to file

    :param grid [rows x cols] matrix: Grid to write. Assumed to already be discretized
    :param filename str: File to save to
    """
    characters = string.ascii_uppercase + "1234"
    tokens = torch.argmax(grid, dim=1).tolist()

    size = int(grid.size(0) ** 0.5)
    assert size ** 2 == grid.size(0) == grid.size(1)

    # split up into 2d coordinates
    tokens = list(map(lambda x: divmod(x, size), tokens))
    # flatten
    tokens = [i for pair in tokens for i in pair]
    # obtain character representation
    tokens = "".join(list(map(lambda x: characters[x], tokens)))
    # split into pairs
    tokens = [tokens[i : i + 2] for i in range(0, len(tokens), 2)]
    # split into Size x Size list
    tokens = [tokens[i : i + size] for i in range(0, len(tokens), size)]

    result = ""

    for i, row in enumerate(tokens):
        if i:
            result += ",\n"
        result += "("
        for j, col in enumerate(row):
            if j:
                result += ", "
            result += col
        result += ")"

    with open(filename, "w") as file:
        file.write(result)
    return result


# Command-line argument parsing
import argparse

parser = argparse.ArgumentParser(description="Reversing Nearness via Gradient Descent")
parser.add_argument(
    "--n_optim", default=10, type=int, help="Number of optimization steps"
)
parser.add_argument(
    "--n_ras", default=5, type=int, help="Number of Sinkhorn-Knopp iterations"
)
parser.add_argument("--size", default=5, type=int, help="Size of the grid")

args = parser.parse_args()

save_name = "size" + str(args.size) + "optim" + str(args.n_optim)

writer = SummaryWriter(log_dir=os.path.join("runs", save_name))

if __name__ == "__main__":
    print("Size:", args.size, " n_optim:", args.n_optim)
    size = torch.tensor([args.size, args.size])

    # Intialize Grid
    grid = Grid(size)
    # Perform optimization
    discrete_loss = grid.forward(args.n_optim, args.n_ras)
    with open("results","a") as file:
        file.write(f"{args.size} {args.n_optim} {args.n_ras} "+str(discrete_loss.item())+"\n")
    # Log results
    writer.add_hparams(
        {"size": args.size, "n_optim": args.n_optim, "n_ras": args.n_ras},
        {"discrete_loss": discrete_loss},
    )

