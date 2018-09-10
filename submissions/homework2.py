import math
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

############################################################################################
# Problem 3 code                                                                           #
############################################################################################

class ProblemThree:
    def __init__(self, points, sigma):
        self.points = points
        self.sigma = float(sigma)

    def objective(self, x):
        total = 0
        for i in range(len(self.points)):
            difference = x - self.points[i]
            gaussian = np.exp(-np.sum(np.square(difference)) / (2 * np.square(self.sigma)))
            total += gaussian
        return total
    
    def gradient(self, x):
        total = np.zeros_like(x, 'float')
        for i in range(len(self.points)):
            difference = x - self.points[i]
            gaussian = np.exp(-np.sum(np.square(difference)) / (2 * np.square(self.sigma)))
            total += gaussian * (-1 / np.square(self.sigma)) * difference
        return total

    def hessian(self, x):
        dim = x.shape[0]
        total = np.zeros((dim, dim))
        for i in range(len(self.points)):
            difference = x - self.points[i]
            gaussian = np.exp(-np.sum(np.square(difference)) / (2 * np.square(self.sigma)))
            inverse_of_sigma_squared = 1 / np.square(self.sigma)
            matrix = inverse_of_sigma_squared * np.outer(difference, difference) - np.eye(dim)
            total += gaussian * (inverse_of_sigma_squared) * matrix
        return total

def steepest_ascent(objective_function, variables, num_iter, step_size):
    objective_value = None
    previous_objective_value = None
    objective_values = []
    solutions = [variables]
    for i in range(num_iter):
        objective_value = objective_function.objective(variables)
        if previous_objective_value != None:
            change =  objective_value - previous_objective_value
            if np.abs(change) == 0:
                print("converged on iteration {}".format(i))
                break
            if change < 1e-15:
                step_size = step_size * .9
        previous_objective_value = objective_value
        objective_values.append(objective_value)
        variables = variables + objective_function.gradient(variables) * step_size
        solutions.append(variables)
    return variables, solutions, objective_value, objective_values

def newton(objective_function, variables, num_iter):
    objective_value = None
    previous_objective_value = None
    objective_values = []
    solutions = [variables]
    for i in range(num_iter):
        objective_value = objective_function.objective(variables)
        if previous_objective_value != None:
            change = previous_objective_value - objective_value
            if np.abs(change) == 0:
                print("converged on iteration {}".format(i))
                break
        previous_objective_value = objective_value
        objective_values.append(objective_value)
        if type(variables) is np.ndarray:
            variables = variables - np.dot(np.linalg.inv(objective_function.hessian(variables) + np.eye(variables.shape[0]) * 1e-8), objective_function.gradient(variables)) 
        else:
            variables = variables - objective_function.gradient(variables) / objective_function.hessian(variables)
        solutions.append(variables) 
    return variables, solutions, objective_value, objective_values

############################################################################################
# Problem 1 and 2 code                                                                     #
############################################################################################

def problem1_objective(p, p_rest, edges, rotation_matrices, handles, handle_indices, lamb, num_points):
    objective_value = 0
    for i in range(num_points):
        for j in edges[i]:
            objective_value += np.sum(np.square(np.dot(rotation_matrices[i], np.transpose(p_rest[i, :] - p_rest[j, :])) - np.transpose(p[i, :] - p[j, :])))

    for index, handle_index in enumerate(handle_indices):
        objective_value += lamb * np.sum(np.square(p[handle_index, :] - handles[index, :]))

    return objective_value

def compute_skew_matrix(c):
    x, y, z = c[0], c[1], c[2]
    skew = np.zeros((3,3))
    skew[0, 1] = -z
    skew[0, 2] = y
    skew[1, 0] = z
    skew[1, 2] = -x
    skew[2, 0] = -y
    skew[2, 1] = x
    return skew

def compute_rotation_matrix(c):
    theta = np.linalg.norm(c)
    skew = compute_skew_matrix(c)
    if theta == 0:
        return np.eye(3)
    return np.eye(3) + (1 - math.cos(theta)) / theta ** 2 * np.dot(skew, skew) + math.sin(theta) / theta * skew

# Derivative of skew symmetric matrix formed by c with respect to c_x
def dc_x_dcx():
    matrix = np.zeros((3,3))
    matrix[1, 2] = -1
    matrix[2, 1] = 1
    return matrix 

# Derivative of skew symmetric matrix formed by c with respect to c_y
def dc_x_dcy():
    matrix = np.zeros((3,3))
    matrix[0, 2] = 1
    matrix[2, 0] = -1
    return matrix

# Derivative of skew symmetric matrix formed by c with respect to c_z
def dc_x_dcz():
    matrix = np.zeros((3,3))
    matrix[0, 1] = -1
    matrix[1, 0] = 1
    return matrix

# Derivative of skew symmetric matrix formed by c squared with respect to c_x
def dc_x_squared_dcx(c):
    x, y, z = c[0], c[1], c[2]
    matrix = np.zeros((3,3))
    matrix[0, 1] = y
    matrix[0, 2] = z
    matrix[1, 0] = y
    matrix[1, 1] = -2*x
    matrix[2, 0] = z
    matrix[2, 2] = -2*x
    return matrix

# Derivative of skew symmetric matrix formed by c squared with respect to c_y
def dc_x_squared_dcy(c):
    x, y, z = c[0], c[1], c[2]
    matrix = np.zeros((3,3))
    matrix[0, 0] = -2*y
    matrix[0, 1] = x
    matrix[1, 0] = x
    matrix[1, 2] = z
    matrix[2, 1] = z
    matrix[2, 2] = -2*y
    return matrix

# Derivative of skew symmetric matrix formed by c squared with respect to c_z
def dc_x_squared_dcz(c):
    x, y, z = c[0], c[1], c[2]
    matrix = np.zeros((3,3))
    matrix[0, 0] = -2*z
    matrix[0, 2] = x
    matrix[1, 1] = -2*z
    matrix[1, 2] = y
    matrix[2, 0] = x
    matrix[2, 1] = y
    return matrix

# Derivative of a rotation matrix with respect to c_i where i can be c_x, c_y, or c_z. 
# The variable index is i.
def derivative_of_R_with_respect_to_ci(c, index):
    x, y, z = c[0], c[1], c[2]
    ci = c[index]
    skew = compute_skew_matrix(c)
    theta = np.linalg.norm(c)
    dtheta_dci = ci / theta
    dcos_theta_term = (math.sin(theta) * theta - 2 + 2 * math.cos(theta)) / (theta ** 3)
    dcos_theta_term = dcos_theta_term * dtheta_dci
    dsin_theta_term = (theta * math.cos(theta) - math.sin(theta)) / (theta ** 2)
    dsin_theta_term = dsin_theta_term * dtheta_dci
    dR_dci = dcos_theta_term * np.dot(skew, skew)
    if index == 0:
        dcx_squared_dci = dc_x_squared_dcx(c)
    elif index == 1:
        dcx_squared_dci = dc_x_squared_dcy(c)
    else:
        dcx_squared_dci = dc_x_squared_dcz(c) 
    dR_dci += ((1 - math.cos(theta)) / (theta ** 2)) * dcx_squared_dci
    dR_dci += dsin_theta_term * skew
    if index == 0:
        dcx_dci = dc_x_dcx()
    elif index == 1:
        dcx_dci = dc_x_dcy()
    else:
        dcx_dci = dc_x_dcz()
    dR_dci += (math.sin(theta) / theta) * dcx_dci
    return dR_dci

def df_dRi(i, neighbor_list, p, p_rest):
    total = 0
    for j in neighbor_list:
        total += np.outer(p[i, :] - p[j, :], p_rest[i, :] - p_rest[j, :])
    return -2 * total

# Just one of many cs
def df_dc_k(i, neighbor_list, p, p_rest, c):
    current_df_dRi = df_dRi(i, neighbor_list, p, p_rest)
    current_df_dc = np.zeros((3,))
    for index in range(3):
        current_df_dc[index] = np.sum(np.multiply(current_df_dRi, derivative_of_R_with_respect_to_ci(c, index)))
    return current_df_dc

# Just one of many ps
def df_dp_k(i, neighbor_list, p, p_rest, rotation_matrices, handles, handle_indices, lamb):
    derivative = np.zeros((3,))
    for j in neighbor_list:
        derivative += -2 * np.dot(rotation_matrices[i], np.transpose(p_rest[i, :] - p_rest[j, :]))
        derivative += 2 * np.transpose(p[i, :])
        derivative += -2 * np.transpose(p[j, :])

    for j in neighbor_list:
        derivative += -2 * np.dot(rotation_matrices[j], np.transpose(p_rest[i, :] - p_rest[j, :]))
        derivative += 2 * np.transpose(p[i, :])
        derivative += -2 * np.transpose(p[j, :])

    if i in handle_indices:
        index = handle_indices.index(i)
        derivative += lamb * 2 * np.transpose(p[i, :] - handles[index, :])
    return derivative

def is_valid_point(point, grid_length):
    x, y, _ = point
    return x < grid_length and x >= 0 and y < grid_length and y >= 0

def generate_mesh_grid(grid_length):
    edges = dict()
    points = np.zeros((grid_length * grid_length, 3))
    handles_points = [(0, 0, 0), (0, grid_length - 1, 0), 
                (int(grid_length // 2), int(grid_length // 2), 0),
               (grid_length - 1, 0, 0), (grid_length - 1, grid_length - 1, 0),
               ]
    handle_indices = []
    handles = np.array([(0, 0, -1), (0, grid_length - 1, -1), 
                (int(grid_length // 2), int(grid_length // 2), 1),
               (grid_length - 1, 0, -1), (grid_length - 1, grid_length - 1, -1),
               ]) 
    for i in range(grid_length):
        for j in range(grid_length):
            index = i * grid_length + j
            point = (i, j, 0)
            if point in handles_points:
                handle_indices.append(index)
            neighbors = []
            above_neighbor = (i, j - 1, 0)
            above_index = i * grid_length + j - 1
            left_neighbor = (i - 1, j, 0)
            left_index = (i - 1) * grid_length + j
            diagonal_neighbor = (i - 1, j - 1, 0)
            diagonal_index = (i-1) * grid_length + j - 1
            if is_valid_point(above_neighbor, grid_length):
                edges[above_index].append(index)
                neighbors.append(above_index)
            if is_valid_point(left_neighbor, grid_length):
                edges[left_index].append(index)
                neighbors.append(left_index)
            if is_valid_point(diagonal_neighbor, grid_length):
                edges[diagonal_index].append(index)
                neighbors.append(diagonal_index)
            edges[index] = neighbors
            points[index, :] = np.array(point)
    return points, edges, handles, handle_indices 

def find_minimum_R(p, p_rest, edges, num_points):
    rotation_matrices = []
    for i in range(num_points):
        neighbors = edges[i]
        P = np.zeros((3, len(neighbors)))
        Q = np.zeros((3, len(neighbors)))
        for index, j in enumerate(neighbors):
            P[:, index] = p_rest[i, :] - p_rest[j, :]
            Q[:, index] = p[i, :] - p[j, :]
        U, _, V_transpose = np.linalg.svd(np.dot(P, np.transpose(Q)))
        R = np.dot(np.transpose(V_transpose), np.transpose(U))
        if np.linalg.det(R) == 1:
            rotation_matrices.append(R)
        else:
            rotation_matrices.append(np.dot(np.transpose(V_transpose), np.dot(np.diag([1, 1, -1]), np.transpose(U))))
    return rotation_matrices

def find_minimum_p(p, p_rest, edges, rotation_matrices, handles, handle_indices, lamb, num_points):
    A_matrices = []
    b_matrices = []
    for i in range(num_points):
        neighbors = edges[i]
        A_matrix = np.zeros((3 * len(neighbors), 3 * num_points))
        b_matrix = np.zeros((3 * len(neighbors),))
        for index, j in enumerate(edges[i]):
            A_matrix[index*3:(index+1)*3, i*3:(i+1)*3] = 1 * np.eye(3)
            A_matrix[index*3:(index+1)*3, j*3:(j+1)*3] = -np.eye(3)
            b_matrix[index*3:(index+1)*3] = np.dot(rotation_matrices[i], np.transpose(p_rest[i, :] - p_rest[j, :]))
        A_matrices.append(A_matrix)
        b_matrices.append(b_matrix)

    sqrt_lamb = math.sqrt(lamb)
    A_matrix = np.zeros((3 * len(handle_indices), 3 * num_points))
    b_matrix = np.zeros((3 * len(handle_indices),))
    for index, handle_index in enumerate(handle_indices):
        A_matrix[index*3:(index+1)*3, handle_index*3:(handle_index + 1)*3] = sqrt_lamb * np.eye(3)
        b_matrix[index*3:(index+1)*3] = sqrt_lamb * handles[index, :]
    A_matrices.append(A_matrix)
    b_matrices.append(b_matrix)
    total_A_matrix = np.concatenate(A_matrices)
    total_b_matrix = np.concatenate(b_matrices)
    symmetric = np.dot(np.transpose(total_A_matrix), total_A_matrix)
    points = np.dot(np.linalg.inv(symmetric), np.dot(np.transpose(total_A_matrix), total_b_matrix))
    return np.reshape(points, (num_points, 3))

def alternating_minimization(p_rest, edges, handles, handle_indices, lamb, num_points, num_iter):
    rotation_matrices = []
    for i in range(num_points):
        rotation_matrices.append(np.eye(3))
    p = p_rest.copy()
    previous_objective_value = None
    objective_values = []

    for iter_index in range(num_iter): 
        objective_value = problem1_objective(p, p_rest, edges, rotation_matrices, handles, handle_indices, lamb, num_points)
        print "iteration {} at value of {}".format(iter_index, objective_value)
        if previous_objective_value != None:
            change =  objective_value - previous_objective_value
            if np.abs(change) == 0:
                print("converged on iteration {}".format(i))
                break
        previous_objective_value = objective_value
        objective_values.append(objective_value)
        p = find_minimum_p(p, p_rest, edges, rotation_matrices, handles, handle_indices, lamb, num_points)
        rotation_matrices = find_minimum_R(p, p_rest, edges, num_points)
    return p, objective_values, objective_value

def gradient_descent(p_rest, edges, handles, handle_indices, lamb, num_points, c_step_size, p_step_size, num_iter):
    cs = np.random.rand(num_points, 3)
    for i in range(num_points):
        cs[i, :] = cs[i, :] / np.linalg.norm(cs[i, :])
    p = p_rest.copy()
    previous_objective_value = None
    objective_values = []

    for iter_index in range(num_iter): 
        rotation_matrices = []
        for i in range(num_points):
            rotation_matrices.append(compute_rotation_matrix(cs[i, :]))
        objective_value = problem1_objective(p, p_rest, edges, rotation_matrices, handles, handle_indices, lamb, num_points)
        print "iteration {} with value {}".format(iter_index, objective_value)
        if previous_objective_value != None:
            change =  objective_value - previous_objective_value
            if np.abs(change) == 0:
                print("converged on iteration {}".format(i))
                break
            if change < 1e-17:
                print "Changing step size"
                p_step_size = p_step_size * .99
                c_step_size = c_step_size * .99
        previous_objective_value = objective_value
        objective_values.append(objective_value)
        df_dcs = np.zeros((num_points, 3))
        df_dps = np.zeros((num_points, 3))
        for i in range(num_points):
            df_dcs[i, :] = df_dc_k(i, edges[i], p, p_rest, cs[i, :])
            df_dps[i, :] = df_dp_k(i, edges[i], p, p_rest, rotation_matrices, handles, handle_indices, lamb)

        cs = cs - c_step_size * df_dcs
        p = p - p_step_size * df_dps

    return p, objective_values, objective_value

def plot_mesh(points, edges, handles):
    handles = np.array(handles)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax = fig.gca(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.8, color="green", edgecolors="None", label="mesh")
    for point_index, neighbor_indices in edges.iteritems():
        point = points[point_index]
        for neighbor_index in neighbor_indices:
            neighbor = points[neighbor_index]
            ax.plot([point[0], neighbor[0]], [point[1], neighbor[1]], [point[2], neighbor[2]], alpha=0.8, color="green")
    ax.scatter(handles[:, 0], handles[:, 1], handles[:, 2], alpha=0.8, color="green", edgecolors="None", label="handles")
    plt.show()


def plot_points(points, solutions, final_solution, objective_values, method):
    points = np.array(points)
    solutions = np.array(solutions)
    final_solution = np.expand_dims(final_solution, 0)
    data = [points, final_solution, solutions]
    colors = ("red", "blue", "green")
    groups = ("initial points", "final solution", "solutions")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax = fig.gca(projection='3d')

    for data, color, group in zip(data, colors, groups):
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ax.scatter(x, y, z,  alpha=0.8, c=color, edgecolors="None", s=30, label=group)
    
    ax.plot(solutions[:, 0], solutions[:, 1], solutions[:, 2], color="green")

    plt.legend(loc=2)
    plt.title(method)
    plt.show()

    plot_convergence(objective_values, method)

def plot_convergence(objective_values, method):
    plt.figure()
    plt.plot(objective_values)
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective value')
    plt.title(method)
    plt.show()

def problem3():
    means = [[1, 1, 1], [-5, -3, -2]]
    covs = [np.eye(3) * 3, np.eye(3) * 2]
    num_gaussians = len(means)
    num_points = 30
    points = []
    for i in range(num_points):
        index = np.random.randint(0, num_gaussians)
        point = np.random.multivariate_normal(means[index], covs[index])
        points.append(point)

    sigma = 5
    f = ProblemThree(points, sigma)
    best_solution, best_solutions, best_objective_value, best_objective_values = None, None, -float('inf'), None
    for point in points:
        solution, solutions, objective_value, objective_values = steepest_ascent(f, point, 400, sigma)
        if objective_value > best_objective_value:
            best_solution, best_solutions, best_objective_value, best_objective_values = solution, solutions, objective_value, objective_values
    plot_points(points, best_solutions, best_solution, best_objective_values, "Steepest Ascent")
    print best_solution


    print "\n" * 5
    print "Newton"
    best_solution, best_solutions, best_objective_value, best_objective_values = None, None, -float('inf'), None
    for point in points:
        solution, solutions, objective_value, objective_values = newton(f, point, 400)
        if objective_value > best_objective_value:
            best_solution, best_solutions, best_objective_value, best_objective_values = solution, solutions, objective_value, objective_values
    plot_points(points, best_solutions, best_solution, best_objective_values, "Newton's method")
    print best_solution

def problem1and2():
    grid_length = 20
    points, edges, handles, handle_indices = generate_mesh_grid(grid_length)
    # Plot the starting configuration
    plot_mesh(points, edges, handles)
    lamb = 90
    num_points = grid_length * grid_length
    p, objective_values, objective_value = alternating_minimization(points, edges, handles, handle_indices, lamb, num_points, 100)
    plot_convergence(objective_values, "Alternating Minimization")
    plot_mesh(p, edges, handles)
    p, objective_values, objective_value = gradient_descent(points, edges, handles, handle_indices, lamb, num_points, .01, .01, 1500)
    plot_convergence(objective_values, "Gradient Descent")
    plot_mesh(p, edges, handles)



########################################################################################################
# Debugging code
########################################################################################################
def gradient_check(f, df, x, delta_x, delta):
    finite_diff = f(x + delta_x) - f(x - delta_x)
    print "finite diff before division"
    print finite_diff
    finite_diff = finite_diff / (2 * delta)
    print "finite diff"
    print finite_diff
    derivative = df(x)
    print "derivative"
    print derivative
    return np.linalg.norm(finite_diff - derivative)

# x = np.random.rand(3)
# delta_x = np.zeros(3)
# delta = np.random.rand() * 1e-6
# delta_x[0] = delta
# derivative = lambda c: dc_x_dcx()
# print "\n" * 3
# print gradient_check(compute_skew_matrix, derivative, x, delta_x, delta)


if __name__ == "__main__":
    # Uncomment these lines to run my code
    problem1and2()
    problem3()
