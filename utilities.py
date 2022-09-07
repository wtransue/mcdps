#
# Several small utility commands that perform simple manipulations, format strings, or write things to stdout
#

import logging
import multiprocessing as mp
from pyswarms.utils.reporter import Reporter
import numpy as np
import ast
from time import time
from functools import partial

numeric = (int, float, np.int16, np.int32, np.int64, np.uint16, np.uint32,
           np.uint64, np.float16, np.float32, np.float64, np.float128)


def lower(x):
    return x.lower() if isinstance(x, str) else x


def fraction_to_float(expr: str) -> float:
    try:
        return float(expr)
    except ValueError:
        numerator, denominator = expr.split('/')
        return float(numerator) / float(denominator)


def sorting_transform(matrix):
    """Returns a matrix `transform' such that `transform @ mat' yields a rearranged matrix equation"""
    #
    # 0. append an identity matrix to keep track of operations
    dim = matrix.shape[0]
    mat = np.hstack((matrix, np.eye(dim)))
    # 1. reduce matrix (akin to red row echelon)
    for i in range(dim):
        pos = np.argmax(abs(mat[i, :-dim]))
        if np.linalg.norm(mat[i, :-dim]) / np.linalg.norm(mat[i, -dim:]) < 1.0e-3:
            continue
        for j in range(dim):
            if i == j:
                mat[j] = (mat[j, pos] / abs(mat[j, pos])) * mat[j]  # make positive
            else:
                mat[j] = mat[j] - (mat[j, pos] / mat[i, pos]) * mat[i]
    # 2. normalize rows (just to standardize things)
    for i in range(dim):
        if max(mat[i, :-dim], key=abs) != 0:
            mat[i] = mat[i] / max(mat[i, :-dim], key=abs)
        else:
            mat[i] = 0 * mat[i]
    # 3. rearrange (to bring closer to [Myz, Mzx, Mxy(, B)] order)
    rearrange = []
    for x in [np.argmax(abs(col)) for col in mat.T[:-dim]]:
        if x not in rearrange:
            rearrange.append(x)
    for x in range(dim):
        if x not in rearrange:
            rearrange.append(x)
    mat = mat[rearrange]
    # 4. return the transformation matrix (previously the identity matrix)
    return mat[:, -dim:]


def express_duration(then, separator='\n'):
    """Reports how much time has elapsed since `then` in days/h/min/sec"""
    now = time()
    diff = now - then
    out = ""
    if diff >= 86400:
        out += str(int(diff // 86400)) + " days "
        diff = diff % 86400
    if diff >= 3600:
        out += str(int(diff // 3600)) + " h "
        diff = diff % 3600
    if diff >= 60:
        out += str(int(diff // 60)) + " min "
        diff = diff % 60
    out += ('%.3f' % diff) + " sec"
    print(separator + 'Job successfully completed in', out, flush=True)


def sf(x, sig=6, cutoff=1e-10):
    """Formats numbers using significant figures. Converts a number 'x' into a string with 'sig' significant figures
    and rounds anything less than `cutoff` (default: 1e-10) in magnitude to zero"""
    if isinstance(x, (list, np.ndarray)):  # if list
        return '[' + ', '.join([sf(n, sig) for n in x]) + ']'
    elif isinstance(x, numeric):  # if valid number
        if np.isnan(x):  # if NaN, return 'NaN' string
            return 'NaN'
        elif x == np.inf:
            return 'Infinity'
        elif x == -np.inf:
            return '-Infinity'
        x = float(x)
    else:  # if invalid 'x', use Python's default str()
        return str(x)
    if not isinstance(sig, int):  # if invalid 'sig', use Python's default str()
        return str(x)
    elif sig <= 0:
        return str(x)
    if abs(x) < cutoff:  # if -1e-10 < x < 1e-10, report as 0
        n = ''.zfill(sig)
        return n[:1] + '.' + n[1:]
    elif np.sign(x) < 0:  # if 'x' is negative
        return '-' + sf(abs(x), sig)
    else:  # x is positive and non-negligible, so we will format...
        q = int(np.floor(np.log10(x)))
        n = str(round(x * 10 ** (sig - 1 - q)))  # only keeping 'sig' sig figs
        if len(n) != sig:  # if rounding changes # digits
            x = round(x * 10 ** (sig - 1 - q))
            x = x * 10 ** (q - sig + 1)
            return sf(x, sig)
        if q > 5 or q < -5:  # use engineering notation if <1e-5 or >=1e6
            n = sf(x * 10 ** (-q), sig)
            if q > 0:
                n += 'e+' + str(q)
            else:
                n += 'e' + str(q)
            return n
        else:  # do not use engineering notation
            if q + 1 >= sig:
                n = n.ljust(q + 1, '0')  # add appropriate number of 0s if missing
            elif q >= 0 and q + 1 < sig:
                n = n[:q + 1] + '.' + n[q + 1:]
            else:
                n = '0.'.ljust(1 - q, '0') + n
            return n


def print_matrix_eq(*list_of_matrices, indent=6):
    """Prints a matrix equation, accepting a list of matrices and non-matrix strings (like '=')"""
    formatted = []
    for mat in list_of_matrices:
        if isinstance(mat, (list, np.ndarray)):
            formatted.append(['['] * len(mat))  # add one '[' per row
            for col in np.transpose(mat):
                formatted.append([sf(x) for x in col])
                formatted.append([' '] * len(col))  # add space after each element
            formatted.pop()  # remove last space
            formatted.append([']'] * len(mat))  # add one ']' per row
        elif isinstance(mat, str):
            formatted.append([mat])  # if it's a string, just add as-is
    n_rows = max([len(x) for x in formatted])  # largest amount of rows
    output = []
    for mat in formatted:
        x = [sf(y) for y in mat]  # format appropriately using sig figs
        x = x + [''] * (n_rows - len(x))  # add empty elements as needed
        max_length = max([len(y) for y in x])  # longest string in column
        x = [' ' * (max_length - len(y)) + y for y in x]  # adjust all entries to this length
        output.append(x)
    output = list(map(list, zip(*output)))  # transpose
    for row in output:
        print(' ' * indent + ''.join(row))


def ordinal(n: int) -> str:
    """Takes an integer and return a string '1st', '2nd', '3rd', '4th', etc."""
    if not isinstance(n, int):
        return str(n)
    else:
        return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(n, 'th')


def spin_to_fraction(spin: (float, int)) -> str:
    if spin % 1 == 0:  # if it's an integer spin
        return str(int(spin))
    elif spin % 1 == 0.5:  # if it's a half-integer spin
        return str(int(2 * spin)) + '/2'
    else:  # otherwise, give up
        return str(spin)


def shape_as(flat: (list, np.ndarray), target_shape: (list, np.ndarray)) -> list:
    """Formats a flat list `flat` into the same shape as irregularly shaped list `target_shape`. If `flat` has too few
    elements to organize in the shape of `target_shape`, raises IndexError"""
    l = []
    for i in range(len(target_shape)):
        if isinstance(target_shape[i], (list, np.ndarray)):
            l.append(shape_as(flat[:len(target_shape[i])], target_shape[i]))
            flat = flat[len(target_shape[i]):]
        else:
            l.append(flat[0])
            flat = target_shape[1:]
    return l


def flatten_shape(nested_list) -> list:
    """Recursively flattens a nested list (or any iterable)"""
    try:
        l = []
        for item in nested_list:
            l.extend(flatten_shape(item))
    except TypeError:
        l = [nested_list]
    return l


def find_variables(expr):
    variables = []
    if isinstance(expr, (list, tuple, set, np.ndarray)):
        for v in [var for item in expr for var in find_variables(item)]:
            if v not in variables:
                variables.append(v)
        return variables
    else:
        a = ast.parse(str(expr))
        for item in ast.walk(a):
            if type(item) == ast.Name:  # if we've found a variable name, add it to the list
                if item.id not in variables and item.id not in ['min', 'max', 'abs', 'round', 'int', 'bool',
                                                                'sin', 'cos', 'tan', 'exp']:
                    variables.append(item.id)
        return variables


class HookeAndJeeves:
    def __init__(self, start, bounds=None, rho=0.7, eps=1.0e-5, itermax=None):
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.dimensions = len(start)
        self.start_point = np.array(start)
        self.bounds = bounds
        if 0 < rho < 1:
            self.rho = rho
        else:
            print('Invalid value of rho is ignored; taking rho = 0.7')
            self.rho = 0.7
        self.eps, self.itermax = eps, itermax

    def optimize(self, objective_func, covariance, n_processes=None, verbose=True, **kwargs):
        def validate_position(pos):
            if self.bounds is None:
                return True
            else:
                return all(self.bounds[0, v] <= pos[v] <= self.bounds[1, v]
                           for v in range(len(self.bounds[0])))

        def all_decreasing(values):
            if len(values) < 2:
                return False
            else:
                return all([values[i] - values[i - 1] < 0 for i in range(1, len(values))])

        iter, d = 0, 0
        current_pos = np.array(self.start_point)
        current_min = min(objective_func([current_pos], **kwargs))
        v_names = kwargs['variable_system'].names

        # Initialize logging
        log_level = logging.INFO if verbose else logging.NOTSET
        self.rep.log(f'Obj. func. args: {kwargs}', lvl=logging.DEBUG)
        self.rep.log(f'Optimize for up to {self.itermax} iters with rho {self.rho}, eps {self.eps}', lvl=log_level)

        # Initialize multiple processes
        pool = None if n_processes is None else mp.Pool(n_processes)

        # Calculating sensitivity to each variable
        directions = np.zeros((2 * self.dimensions, self.dimensions))
        for n in range(self.dimensions):
            displace = self.bounds[1, n] - self.bounds[0, n]
            displace = max(min(covariance[n, n] ** 0.5 / 2, displace / 10), displace / 100)
            directions[2 * n, n], directions[2 * n + 1, n] = displace, -displace
        self.rep.log(f'Displacements: {directions}')

        while self.rho ** d > self.eps and iter < self.itermax:
            scale = self.rho ** d * (0.95 + 0.1 * np.random.rand())  # provide a little noisiness
            if iter % 10 == 0:
                self.rep.log(f'Iteration {iter} cost {current_min} rho {self.rho ** d} position {current_pos}',
                             lvl=log_level)

            iter += 1
            positions = np.array([current_pos + scale * direction for direction in directions])
            if pool is None:
                results = objective_func(positions, **kwargs)
            else:
                results = np.concatenate(pool.map(partial(objective_func, **kwargs),
                                                  np.array_split(positions, pool._processes)))
            if self.bounds is not None:
                for u in range(2 * self.dimensions):
                    if not validate_position(positions[u]):
                        results[u] = np.inf  # if out of bounds, set result to infinity

            if all(x > current_min for x in results):
                d += 1  # shrink search radius if we haven't found a minimizing direction
                self.rep.log(f'{iter}: Contracting rho to {self.rho ** d} after '
                             f'getting {min(x - current_min for x in results)}', lvl=log_level)
                continue

            n_moves = 100  # protect against infinite loops
            i = np.argmin(results)
            direction, results = scale * directions[i], [current_min]
            self.rep.log(f'{iter}: Changing {v_names[i // 2]} in steps of {direction[i // 2]}')
            while len(results) <= n_moves:
                if pool is None:
                    new_position = current_pos + direction * len(results)
                    if validate_position(new_position):
                        results.extend(objective_func([new_position], **kwargs))
                    else:
                        results.append(np.inf)
                else:
                    new_position = np.array([current_pos + direction * (len(results) + j)
                                             for j in range(pool._processes)])
                    new_results = np.concatenate(pool.map(partial(objective_func, **kwargs),
                                                          np.array_split(new_position, pool._processes)))
                    for j in range(len(new_position)):
                        if not validate_position(new_position[j]):
                            new_results[j] = np.inf
                    results.extend(new_results)
                if not all_decreasing(results):  # if error is increasing (or out-of-bounds), break
                    break

            old_position = current_pos
            for j in range(1, len(results)):
                j_result, j_position = results[j], old_position + direction * j
                self.rep.log(f'{iter}: Testing {j_result} at {j_position} ({validate_position(j_position)})',
                             lvl=log_level)
                if j_result >= current_min or not validate_position(j_position):
                    break
                current_min, current_pos = j_result, j_position
            self.rep.log(f'{iter}: Moving best guess to give {current_min} at {current_pos}', lvl=log_level)

        if pool is not None:
            pool.close()
        self.rep.log(f'Optimization finished after {iter} steps: cost {current_min} position {current_pos}',
                     lvl=log_level)
        return current_min, current_pos


class GridSearch:
    def __init__(self, divisions, bounds, start=0):
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.start = start
        self.bounds = bounds
        self.divisions = divisions

    def optimize(self, objective_func, n_processes=None, verbose=True, **kwargs):
        def resolve(number):
            def make_generator(n):
                r, i = int(n), 0
                while r > 0:
                    yield r % self.divisions[i]
                    r //= self.divisions[i]
                    i += 1
            return [k for k in make_generator(number)]

        def partition(start, end, p=1):
            i = int(start)
            while i < end:
                yield range(i, min(end, i + p))
                i += p

        def get_grid_values(indices):
            return [grid_values[i][indices[i]] for i in range(len(indices))]

        # Initialize logging
        log_level = logging.INFO if verbose else logging.NOTSET

        # Initialize multiple processes
        proc = 1 if n_processes is None else n_processes

        # Formulate grid
        if isinstance(self.divisions, int):
            self.divisions = [self.divisions] * self.bounds.shape[1]
        grid_values = [np.linspace(self.bounds[0, b], self.bounds[1, b], self.divisions[b])
                       for b in range(self.bounds.shape[1])]
        n_vars = len(grid_values)
        self.rep.log(f'{np.prod(self.divisions)} iterations across grid divisions:\n{grid_values}', lvl=log_level)
        best_value, best_position = np.inf, None
        for grouping in partition(self.start, np.prod(self.divisions), proc):
            with mp.Pool(n_processes) as pool:
                positions = [resolve(n) for n in grouping]
                positions = [get_grid_values(r + [0] * (n_vars - len(r))) for r in positions]

                results = [pool.apply_async(partial(objective_func, **kwargs), ([p],)) for p in positions]
                results = [a.get(timeout = 120) for a in results]  # don't want longer than 2 mins for a single pt

                for n in range(len(positions)):
                    self.rep.log(f'Iter {grouping[n]} position {positions[n]} yielded {results[n]}', lvl=log_level)
                    if results[n][0] < best_value:
                        self.rep.log(f'New best position', lvl=log_level)
                        best_value, best_position = results[n], positions[n]

        self.rep.log(f'Optimization finished with cost {best_value} at position {best_position}',
                     lvl=log_level)
        return best_value, best_position
