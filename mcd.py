import numpy as np
from yaml import safe_load
from scipy.linalg import block_diag
from time import time
import sys
import csv
import pyswarms as ps
from utilities import *
from hamiltonian import SpinSystem, IntegrationGrid, integrate


def reduce_dimension(mat, weight, yvals, cutoff):
    if cutoff == 0 or cutoff >= 1:
        return mat, np.eye(mat.shape[1])
    else:
        n = len(mat)
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        us = u @ np.diag(s)
        if cutoff > 0:
            # if any singular values are noisily small, we'll clip; min 1
            dims = max(sum(int(i > cutoff) for i in (s / max(s))), 1)
            reduced_chi_sq = [np.inf]
            for p in range(1, dims + 1):
                us = u[:, :p] @ np.diag(s[:p])
                params = us.T * weight
                params = np.linalg.pinv(params @ us) @ (params @ yvals)
                residuals = us @ params - yvals
                reduced_chi_sq.append((residuals * weight) @ residuals / (n - p))
            dims = np.argmin(reduced_chi_sq)  # get dimensionality with smallest reduced chi**2 value
        else:
            dims = max(int(abs(cutoff)), 1)
        return us[:, :dims], vt[:dims]


class Simulation:
    def __init__(self, fitted, simulations, params, transform):
        self.fitted = fitted
        self.simulations = simulations
        self.parameters = params
        self.transform = transform

    @property
    def adjustment(self):
        return max(self.simulations.shape) / (max(self.simulations.shape) - min(self.simulations.shape))


class Dataset:
    def __init__(self, input=None, output=None):
        if input is None or isinstance(input, str):  # initialize `public` variables
            self.tempfield = None
            self.filename, self.header, self.output = input, 0, output
            self._data, self._error, self.centers, self._num = np.array([]), np.array([]), [], 0
            self._polarizations, self._constraints, self._b_term, self._weights = None, [], [], None
            if input is not None:
                self.read_file()
        elif isinstance(input, Dataset):  # copy variables from supplied Dataset, if present
            self.tempfield = np.array(input.tempfield)
            self.filename = str(input.filename)
            self.header = int(input.header)
            self.output = output if output is not None else str(input.output)
            self._data = np.array(input.data)
            self._error = np.array(input.error)
            self.centers = list(input.centers)
            self._polarizations = list(input.polarizations)
            self._constraints = list(input.constraints)
            self._b_term = list(input.b_term)
            self._weights = np.array(input._weights)
        else:
            raise TypeError(f'Expected `input` as a str or Dataset object not {type(input)}')
        self.spin_system, self.integration_grid, self.cutoff = None, None, None
        self.simulation = np.zeros(self.data.shape)  # DEBUG
        self.wrss = np.zeros(self.data.shape[0])  # DEBUG
        self.cartesian, self.svd_reduced, self.svd_transform, self.svd_polarizations = None, None, None, None

    @property
    def num(self):
        self._num = max(len(self._data) if isinstance(self._data, (list, np.ndarray)) else 0,
                        len(self.polarizations) if isinstance(self.polarizations, (list, np.ndarray)) else 0)
        return self._num

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data_array: (list, np.ndarray)):
        self._data = np.array(data_array)
        self.wrss = np.zeros(self._data.shape[0])
        if not isinstance(self.centers, (list, np.ndarray)):
            self.centers = [0] * self.num
        elif len(self.centers) < self.num:
            self.centers.extend([0] * (self.num - len(self.centers)))
        if not isinstance(self._b_term, list):
            self._b_term = [None] * self.num
        elif len(self._b_term) < self.num:
            self._b_term.extend([None] * (self.num - len(self._b_term)))
        if not isinstance(self._constraints, (list, np.ndarray)):
            self._constraints = [None] * self.num
        elif len(self._constraints) < self._data.shape[0]:
            self._constraints.extend([None] * (self.num - len(self._constraints)))
        if not isinstance(self._error, np.ndarray):
            self._error = np.zeros(self._data.shape)
        if self._error.shape != self._data.shape:
            self._error = np.zeros(self._data.shape)
            self._error = self._error[:self._data.shape[0], :self._data.shape[1]]
            padding = ((0, self._data.shape[0] - self._error.shape[0]), (0, self._data.shape[1] - self._error.shape[1]))
            self._error = np.pad(self._error, padding)
        self.simulation = np.zeros((self.num, len(self.tempfield)))

    @property
    def error(self) -> np.ndarray:
        return self._error

    @error.setter
    def error(self, error_array: (list, np.ndarray)):
        self._error = np.array(error_array)
        # weights are reciprocal variance unless zero if found (because 0^-2 is infinite)
        self._weights = np.array([r ** -2 if (0 not in r) else np.ones(len(r)) for r in self.error])
        self._weights = np.array([r / r.sum() for r in self._weights])
        if not isinstance(self._data, np.ndarray):
            self._data = np.zeros(self._error.shape)
        if np.shape(error_array) != self._data.shape:
            self._data = self._data[:self._error.shape[0], :self._error.shape[1]]
            padding = ((0, self._error.shape[0] - self._data.shape[0]), (0, self._error.shape[1] - self._data.shape[1]))
            self._data = np.pad(self._data, padding)
            self.simulation = np.zeros((self.num, len(self.tempfield)))
            self.wrss = np.zeros(self._data.shape[0])
        if not isinstance(self.centers, (list, np.ndarray)):
            self.centers = [0] * self._data.shape[0]
        elif len(self.centers) < self.num:
            self.centers.extend([0] * (self.num - len(self.centers)))
        if not isinstance(self._b_term, list):
            self._b_term = [None] * self.num
        elif len(self._b_term) < self.num:
            self._b_term.extend([None] * (self.num - len(self._b_term)))
        if not isinstance(self._constraints, (list, np.ndarray)):
            self._constraints = [None] * self.num
        elif len(self._constraints) < self._data.shape[0]:
            self._constraints.extend([None] * (self.num - len(self._constraints)))
        if not isinstance(self._error, np.ndarray):
            self._error = np.zeros(self._data.shape)

    @property
    def weights(self):
        return self._weights

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list=None):
        self._constraints = [None] * self._data.shape[0]
        if constraint_list is None:
            constraint_list = [None] * self._data.shape[0]
        if len(constraint_list) != self._data.shape[0]:
            raise AttributeError('Number of constraints must equal number of transitions in data set')
        for i in range(self._data.shape[0]):
            if constraint_list[i] is None:
                self._constraints[i] = None
            elif isinstance(constraint_list[i], (list, np.ndarray)):
                c = np.array(constraint_list[i])
                if len(c.shape) == 1:
                    c = c[np.newaxis, :]
                if c.shape[1] not in [3, 9]:
                    raise AttributeError(f'Uninterpretable constraint has dimension {c.shape}')
                self._constraints[i] = c
            else:
                raise TypeError(f'Polarization constraint of unrecognized type {type(constraint_list[i])}')

    @property
    def polarizations(self):
        return self._polarizations

    @polarizations.setter
    def polarizations(self, polarization_array):
        self._polarizations = polarization_array
        if not isinstance(self.centers, (list, np.ndarray)):
            self.centers = [0] * self.num
        elif len(self.centers) < self.num:
            self.centers.extend([0] * (self.num - len(self.centers)))
        if not isinstance(self._b_term, list):
            self._b_term = [None] * self.num
        elif len(self._b_term) < self.num:
            self._b_term.extend([None] * (self.num - len(self._b_term)))
        self.simulation = np.zeros((self.num, len(self.tempfield)))

    @property
    def b_term(self):
        return self._b_term

    @b_term.setter
    def b_term(self, b_array: list):
        if len(b_array) != self.num:
            raise AttributeError(f'B term intensity (or lack thereof) must be provided for each transition')
        self._b_term = b_array

    def read_file(self, filename: str = None) -> None:
        """Attempts to read file `filename` and returns the number of rows of data found in the file. Raises
        RuntimeError if the file cannot be opened or cannot be parsed."""
        if filename is None:
            filename = self.filename
        header = 0
        while True:
            try:
                data = np.loadtxt(filename, delimiter=',', skiprows=header)
            except ValueError:
                header += 1
                continue
            break
        if data.shape[0] == 0:
            raise RuntimeError(f'File {filename} contained no readable data')
        if data.shape[1] < 2:
            raise RuntimeError(f'Incorrectly formatted file {filename} has too few columns')
        if data.shape[1] % 2 == 1:
            raise RuntimeError(f'Incorrectly formatted file {filename} has an odd number of columns')
        self.filename, self.header = filename, header
        self.tempfield = data[:, :2].astype(float)
        if not all(self.tempfield[:, 0] > 0):
            temps = self.tempfield[:, 0]
            raise ValueError('All temperatures must be absolute and must therefore be positive '
                             f'but nonpositive values were found: {temps[temps <= 0]}')
        data = data[:, 2:].T
        self.data = data[0::2]  # every other row is data; if no data supplied, returns a (0, n)-shaped array
        self.error = data[1::2]  # interspersed by error; if no data supplied, returns a (0, n)-shaped array
        if self.centers is None:
            self.centers = [0] * self.num
        if self.constraints is None:
            self.constraints = [None] * self.num

    def write_file(self, filename=None) -> None:
        """Writes a CSV file containing [T, H, Data, Error, Simulation, Data, Error, Simulation, ...]"""
        if filename is None:
            filename = self.output
        num = self.num
        header = ['Temp', 'Field', 'bH/2kT']
        for i in range(num):
            header.extend([f'Sim {i + 1}', f'Data {i + 1}', f'Error {i + 1}'])
        output = np.transpose(self.tempfield)
        # (Bohr mag)/(2kB) = 0.33585690476000785 Kelvin/Tesla
        output = np.append(output, [0.33585690476000785 * output[1] / output[0]], axis=0)
        for i in range(num):
            if i < self.simulation.shape[0]:
                output = np.append(output, self.simulation[np.newaxis, i], axis=0)
            else:
                output = np.append(output, np.zeros((1, len(self.tempfield))), axis=0)
            if i < len(self.data):
                output = np.append(output, self.data[np.newaxis, i], axis=0)
                output = np.append(output, self.error[np.newaxis, i], axis=0)
            else:
                output = np.append(output, np.zeros((2, len(self.tempfield))), axis=0)
        output = np.append([header], output.T, axis=0)
        with open(filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            for r in output:
                writer.writerow(r)

    def _get_center(self, simulations, i):
        """From a (Cartesian) list of simulations, takes the entry for the correct spin center and applies any
        constraints it may find. Constraints should be stored as (n,3) or (n,9) ndarrays. For example,
        - A constraints along the [1, 0, 0] vector is constraints[i] = np.array([[1, 0, 0]])
        - A constraints within the xy (Myz/Mzx) plane is constraints[i] = np.array([[1, 0, 0], [0, 1, 0]])"""
        if self.constraints[i] is not None:
            return simulations[self.centers[i]] @ self.constraints[i].T
        else:
            return simulations[self.centers[i]]

    def _fit_subroutine(self, simulations, transition, cutoff):
        """Uses weighted linear regression to get closest match to data. In weighted linear regression, the coefficients
        of best fit (c) relating the X matrix to the y vector can be found by c = (X.T W X)^-1 X.T W y using weight
        matrix W. Doing this directly here instead of relying on numpy's built-in regression wrapper should (?) make the
        routine slightly faster."""
        b_offset = np.zeros(len(self.tempfield))
        weight = self._weights[transition]
        x = self._get_center(simulations, transition)
        if self.b_term[transition]:
            if isinstance(self.b_term[transition], numeric):
                b_offset = self.b_term[transition] * self.tempfield[:, 1]
            elif self.b_term[transition] == 'fit':
                x = np.hstack((x, self.tempfield[:, [1]]))
            else:
                raise RuntimeError(f'Unrecognized B-term parameter {self.b_term[transition]}')
        y = self.data[transition] - b_offset
        x, transform = reduce_dimension(x, weight, y, cutoff)  # set cutoff to 0 to skip svd
        params = x.T * weight
        params = np.linalg.inv(params @ x) @ params @ y  # best fit parameters
        return Simulation(x @ params + b_offset, x, params, transform)

    def fit_to(self,
               spin_system: SpinSystem,
               variable_values: np.ndarray,
               integration_grid: IntegrationGrid,
               cutoff) -> list:
        # First we simulate by integration, transposing so that it is formatted as [center, T/H index, intensity]
        simulations = np.array([integrate(temperature, field, spin_system, variable_values, integration_grid)
                                for temperature, field in self.tempfield]).transpose((1, 0, 2))
        return [self._fit_subroutine(simulations, i, cutoff) for i in range(self.data.shape[0])]

    def residuals(self,
                  spin_system: SpinSystem,
                  variable_values: np.ndarray,
                  integration_grid: IntegrationGrid,
                  cutoff) -> np.ndarray:
        data_simulations = self.fit_to(spin_system, variable_values, integration_grid, cutoff)
        return np.array([sim.fitted for sim in data_simulations]) - self.data

    def quick_wrss(self,
                   spin_system: SpinSystem,
                   variable_values: np.ndarray,
                   integration_grid: IntegrationGrid,
                   cutoff, adjust=True) -> float:
        data_simulations = self.fit_to(spin_system, variable_values, integration_grid, cutoff)
        residuals_vectors = np.array([sim.fitted for sim in data_simulations]) - self.data
        wrss = 0
        for k in range(len(residuals_vectors)):
            scale = data_simulations[k].adjustment if adjust else 1  # try to decrease bias
            wrss += scale * (residuals_vectors[k] * self._weights[k]) @ residuals_vectors[k]
        return wrss

    def get_reduced_polarizations(self,
                                  spin_system: SpinSystem,
                                  variable_values: np.ndarray,
                                  integration_grid: IntegrationGrid,
                                  cutoff) -> list:
        """Same routine as `fit_to' but returns the polarizations and transformation matrices."""
        # First we simulate by integration, transposing so that it is formatted as [center, T/H index, intensity]
        data_simulations = self.fit_to(spin_system, variable_values, integration_grid, cutoff)
        return [[sim.parameters, sim.transform] for sim in data_simulations]

    def get_reduced_polarization_derivatives(self,
                                             spin_system: SpinSystem,
                                             variable_values: np.ndarray,
                                             integration_grid: IntegrationGrid,
                                             cutoff):
        data_simulations = self.fit_to(spin_system, variable_values, integration_grid, cutoff)
        return block_diag(*[sim.simulations for sim in data_simulations])

    def calculate_given(self, variable_values: np.ndarray = None):
        """Uses self.polarizations to simulate data for each transition."""
        if variable_values is None:
            variable_values = []
        # First we simulate by integration, transposing so that it is formatted as [center, T/H index, intensity]
        self.cartesian = np.array([integrate(temperature, field,
                                             self.spin_system, variable_values, self.integration_grid)
                                   for temperature, field in self.tempfield]).transpose((1, 0, 2))
        self.simulation = np.zeros((len(self.polarizations), len(self.tempfield)))
        for i in range(len(self.polarizations)):
            self.simulation[i] = self.cartesian[self.centers[i]] @ self.polarizations[i]
            if self.b_term[i]:
                if isinstance(self.b_term[i], numeric):  # if we're given B term, subtract from data
                    self.simulation[i] += self.b_term[i] * self.tempfield[:, 1]
                else:
                    raise TypeError(f'B term must be numeric for simulations')

    def calculate_by_fit(self, variable_values: np.ndarray = None, update=True):
        """Uses weighted linear regression to calculate the WRSS, polarizations, and simulated data for each transition
        in self.data. Updates self to retain simulated data and reduced polarization info."""
        if variable_values is None:
            variable_values = []
        # First we simulate by integration, transposing so that it is formatted as [center, T/H index, intensity]
        cartesian = np.array([integrate(temperature, field, self.spin_system, variable_values, self.integration_grid)
                              for temperature, field in self.tempfield]).transpose((1, 0, 2))
        svd_reduced = []
        svd_transform = []
        svd_polarizations = []
        simulation = np.zeros(self.data.shape)
        wrss = np.zeros(self.data.shape[0])
        for i, y in enumerate(self.data):
            weight = self._weights[i]  # read in weights for the data points
            x = self._get_center(cartesian, i)  # simulation for correct center, constrained if necessary
            if self.b_term[i]:
                if self.b_term[i] == 'fit':  # if we're fitting B term, add to `sim` variable before SVD
                    x = np.hstack((x, self.tempfield[:, [1]]))
                elif isinstance(self.b_term[i], numeric):  # if we're given B term, subtract from data
                    y = y - self.b_term[i] * self.tempfield[:, 1]
            x, transform = reduce_dimension(x, weight, y, self.cutoff)
            params = x.T * weight
            params = np.linalg.inv(params @ x) @ params @ y  # best fit parameters `b`
            svd_reduced.append(x)
            svd_transform.append(transform)
            svd_polarizations.append(params)  # Cartesian by pinv(transform) @ params)
            simulation[i] = x @ params
            wrss[i] = sum((y - x @ params) ** 2 * weight)
        if update:
            self.cartesian = cartesian
            self.svd_reduced = svd_reduced
            self.svd_transform = svd_transform
            self.svd_polarizations = svd_polarizations
            self.simulation = simulation
            self.wrss = wrss
        return wrss

    def perturb_svd_pol_wrss(self, perturbation):
        def add(list1, list2):
            if isinstance(list1, (int, float)):
                return list1 + list2
            else:
                return [add(*i) for i in zip(list1, list2)]

        polarizations = add(self.polarizations, perturbation)
        wrss = np.zeros(self.data.shape[0])
        for i, y in enumerate(self.data):
            weight = self._weights[i]  # read in weights for the data points
            sim = self.svd_reduced[i]
            params = polarizations[i]
            if isinstance(self.b_term[i], numeric):
                y = y - self.b_term[i] * self.tempfield[:, 1]
            wrss[i] = sum((y - sim @ params) ** 2 * weight)
        return wrss


class DataCollection:
    def __init__(self, list_of_datasets=None):
        if list_of_datasets is None:
            list_of_datasets = []
        self.datasets = list_of_datasets
        self._split, self._split_lengths, self._split_directory = None, None, None

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        return iter(self.datasets)

    def get_costs(self, spin_system, variable_values, integration_grid, cutoff, together=True, adjust=True):
        if together:
            return sum(dataset.quick_wrss(spin_system, variable_values, integration_grid, cutoff, adjust=adjust)
                       for dataset in self.datasets)
        else:
            return [dataset.quick_wrss(spin_system, variable_values, integration_grid, cutoff, adjust=adjust)
                    for dataset in self.datasets]

    def get_residuals(self, spin_system, variable_values, integration_grid, cutoff):
        return np.concatenate([np.concatenate(dataset.residuals(spin_system, variable_values, integration_grid, cutoff))
                               for dataset in self.datasets])


class VariableSystem:
    def __init__(self, g_tensor=None, d_tensor=None, exchange_j=None,
                 exchange_g=None, exchange_d=None):
        def to_lambda_expr(expr):
            if isinstance(expr, (list, tuple, np.ndarray)):
                return [to_lambda_expr(item) for item in expr]
            return 'lambda ' + ', '.join(self.names) + ': ' + str(expr)

        if g_tensor is None:
            g_tensor = []
        if d_tensor is None:
            d_tensor = []
        if exchange_j is None:
            exchange_j = []
        if exchange_g is None:
            exchange_g = []
        if exchange_d is None:
            exchange_d = []
        self.inputs = {'g_tensor': g_tensor,
                       'd_tensor': d_tensor,
                       'exchange_j': exchange_j,
                       'exchange_g': exchange_g,
                       'exchange_d': exchange_d}
        self.names = []
        self.names.extend([var for var in find_variables(g_tensor) if var not in self.names])
        self.names.extend([var for var in find_variables(d_tensor) if var not in self.names])
        self.names.extend([var for var in find_variables(exchange_j) if var not in self.names])
        self.names.extend([var for var in find_variables(exchange_g) if var not in self.names])
        self.names.extend([var for var in find_variables(exchange_d) if var not in self.names])
        self.bounds = []
        self.values = []
        self.g_tensor = to_lambda_expr(g_tensor)
        self.d_tensor = to_lambda_expr(d_tensor)
        self.exchange_j = to_lambda_expr(exchange_j)
        self.exchange_g = to_lambda_expr(exchange_g)
        self.exchange_d = to_lambda_expr(exchange_d)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        yield from self.names

    def set_bounds(self, bounds):
        self.bounds = [np.sort(bounds[name]) for name in self.names]

    def set_values(self, values):
        if len(values) != len(self.names):
            raise ValueError('Improper number of variable values to set')
        self.values = list(values)

    def resolve(self, vals=None):
        def from_lambda_expr(expr):
            if isinstance(expr, list):
                return [from_lambda_expr(item) for item in expr]
            sin, cos, tan, exp = np.sin, np.cos, np.tan, np.exp  # hopefully this allows users to use these in eval()
            func = eval(expr)
            return func(*values)

        values = vals if vals is not None else list(self.values)
        if len(values) != len(self.names):
            raise RuntimeError('Variable number mismatch')
        return [from_lambda_expr(self.g_tensor),
                from_lambda_expr(self.d_tensor),
                from_lambda_expr(self.exchange_j),
                from_lambda_expr(self.exchange_g),
                from_lambda_expr(self.exchange_d)]


def cost_at_each_particle(particle_positions, spin_system, variable_system, integration_grid,
                          data_collection, cutoff):
    return [data_collection.get_costs(spin_system, variable_system.resolve(position),
                                      integration_grid, cutoff, together=True)
            for position in particle_positions]


class FitSystem:
    """Performs fitting of the magnetic parameters"""

    def __init__(self, spin_system: SpinSystem, variable_system: VariableSystem,
                 integration_grid: IntegrationGrid, data_collection: DataCollection, settings: dict) -> None:
        """Initializes the FitSystem object"""
        self.spin_system = spin_system
        self.variable_system = variable_system
        self.integration_grid = integration_grid
        self.data_collection = data_collection
        self._split_collection = None
        self.settings = settings
        self.cost = []
        self.history = []

    def fit_no_variables(self):
        self.cost = self.data_collection.get_costs(self.spin_system, self.variable_system.resolve(),
                                                   self.integration_grid, cutoff=settings['cutoff'],
                                                   together=False, adjust=False)

    def particle_swarm_fit(self) -> float:
        """Perform particle swarm optimization. We will also hijack the Pyswarms `_populate_history` function to
        record the current cost at each step. For some reason, this is not built-in; it only records the current best
        at each step."""

        def wrapper(populate_history):
            def _wrap(self, hist):
                if not hasattr(self, 'current_cost_history'):
                    self.current_cost_history = []
                self.current_cost_history.append(self.swarm.current_cost)
                populate_history(self, hist)

            return _wrap

        ps.base.base_single.SwarmOptimizer._populate_history = wrapper(
            ps.base.base_single.SwarmOptimizer._populate_history)
        settings = self.settings
        options = {'c1': settings['cognitive'], 'c2': settings['social'], 'w': settings['inertial']}
        bounds = np.array(self.variable_system.bounds).T
        if settings['parallel'] is not None and settings['parallel'] < 2:
            settings['parallel'] = None
        optimizer = ps.single.GlobalBestPSO(n_particles=settings['particles'],
                                            dimensions=len(self.variable_system),
                                            options=options, bounds=bounds)
        cost, pos = optimizer.optimize(cost_at_each_particle,
                                       iters=settings['iterations'],
                                       n_processes=settings['parallel'],
                                       spin_system=self.spin_system,
                                       variable_system=self.variable_system,
                                       integration_grid=self.integration_grid,
                                       data_collection=self.data_collection,
                                       cutoff=settings['cutoff'])
        self.history = [np.array(optimizer.cost_history),
                        np.array(optimizer.current_cost_history),
                        np.array(optimizer.pos_history),
                        np.array(optimizer.velocity_history)]
        if settings['history']:
            with open(settings['history'] + '.cost.csv', mode='w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([f'Particle {n + 1}' for n in range(settings['particles'])])
                for r in self.history[1]:
                    writer.writerow(r)
            with open(settings['history'] + '.position.csv', mode='w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([f'Particle {n + 1} {variable}' for n in range(settings['particles'])
                                 for variable in self.variable_system])
                for r in self.history[2]:
                    writer.writerow(r.flatten())
            with open(settings['history'] + '.velocity.csv', mode='w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([f'Particle {n + 1} {variable}' for n in range(settings['particles'])
                                 for variable in self.variable_system])
                for r in self.history[3]:
                    writer.writerow(r.flatten())
        self.cost = self.data_collection.get_costs(self.spin_system, self.variable_system.resolve(pos),
                                                   self.integration_grid, cutoff=settings['cutoff'],
                                                   together=False, adjust=False)
        self.variable_system.set_values(pos)
        self.spin_system.set_parameters(*self.variable_system.resolve())
        return cost

    def hooke_and_jeeves_fit(self) -> float:
        """Perform Hooke & Jeeves optimization"""
        settings = self.settings
        bounds = np.array(self.variable_system.bounds).T
        if settings['parallel'] is not None and settings['parallel'] < 2:
            settings['parallel'] = None
        optimizer = HookeAndJeeves(start=self.variable_system.values,
                                   bounds=bounds,
                                   rho=settings['rho'],
                                   eps=settings['eps'],
                                   itermax=settings['iterations'])
        covariance = self.covariance_matrix()
        cost, pos = optimizer.optimize(cost_at_each_particle, covariance,
                                       n_processes=settings['parallel'],
                                       spin_system=self.spin_system,
                                       variable_system=self.variable_system,
                                       integration_grid=self.integration_grid,
                                       data_collection=self.data_collection,
                                       cutoff=settings['cutoff'])
        self.cost = self.data_collection.get_costs(self.spin_system, self.variable_system.resolve(pos),
                                                   self.integration_grid, cutoff=settings['cutoff'],
                                                   together=False, adjust=False)
        self.variable_system.set_values(pos)
        self.spin_system.set_parameters(*self.variable_system.resolve())
        return cost

    def grid_search_fit(self) -> float:
        """Perform Grid Search optimization"""
        settings = self.settings
        bounds = np.array(self.variable_system.bounds).T
        if settings['parallel'] is not None and settings['parallel'] < 2:
            settings['parallel'] = None
        optimizer = GridSearch(settings['divisions'], bounds, settings['start'])
        cost, pos = optimizer.optimize(cost_at_each_particle,
                                       n_processes=settings['parallel'],
                                       spin_system=self.spin_system,
                                       variable_system=self.variable_system,
                                       integration_grid=self.integration_grid,
                                       data_collection=self.data_collection,
                                       cutoff=settings['cutoff'])
        self.cost = self.data_collection.get_costs(self.spin_system, self.variable_system.resolve(pos),
                                                   self.integration_grid, cutoff=settings['cutoff'],
                                                   together=False, adjust=False)
        self.variable_system.set_values(pos)
        self.spin_system.set_parameters(*self.variable_system.resolve())
        return cost

    def polarization_info(self):
        return [dataset.get_reduced_polarizations(self.spin_system,
                                                  self.variable_system.resolve(),
                                                  self.integration_grid,
                                                  self.settings['cutoff'])
                for dataset in self.data_collection]

    def covariance_matrix(self, epsilon=0.01, include_polarizations=True):
        # Weighted covariance matrix:
        # cov = (res @ weight @ res) / (n - p) * inv(jacob.T @ weight @ jacob)
        # cov is covariance matrix of fitted parameters
        # res is a vector of residuals
        # weight is a weight matrix equal to inverse of data covariance matrix
        # jacob is a Jacobian matrix of predicted data vs parameter variable
        best_fit = np.array(self.data_collection.get_residuals(self.spin_system,
                                                               self.variable_system.resolve(),
                                                               self.integration_grid,
                                                               self.settings['cutoff']))
        if len(self.variable_system.values) > 0:
            position = self.variable_system.values
            displacements = epsilon * np.eye(len(self.variable_system))
            jacobian = [self.data_collection.get_residuals(self.spin_system,
                                                           self.variable_system.resolve(position + disp),
                                                           self.integration_grid,
                                                           self.settings['cutoff'])
                        for disp in displacements]
            jacobian = np.array([sim - best_fit for sim in jacobian]) / epsilon
        else:
            jacobian = np.zeros((0, len(best_fit)))
        if include_polarizations:
            pol_derivs = block_diag(*[dataset.get_reduced_polarization_derivatives(self.spin_system,
                                                                                   self.variable_system.resolve(),
                                                                                   self.integration_grid,
                                                                                   self.settings['cutoff'])
                                      for dataset in self.data_collection]).T
            jacobian = np.concatenate((jacobian, pol_derivs))
        weights = flatten_shape([flatten_shape(dataset.weights) for dataset in self.data_collection.datasets])
        covariance = (jacobian * weights) @ jacobian.T
        try:
            covariance = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            print('WARNING: Non-invertible matrix encountered while calculating (co)variance matrix...')
            print('         Switching to Moore-Penrose pseudoinverse; however, results may be meaningless')
            covariance = np.linalg.pinv(covariance)
        covariance *= ((best_fit * weights) @ best_fit) / (len(weights) - len(self.variable_system))
        return covariance


def read_spin_system(yaml, run_type, settings) -> (SpinSystem, VariableSystem):
    def check_shape(mat, shapes):
        # a g tensor can have shapes [1, 2, 3, 6], D tensor [0, 1, 2, 5], exchange G [0, 3], exchange D [0, 1, 2, 5]
        if isinstance(mat, list) and len(mat) in shapes:
            return all(isinstance(x, numeric) or isinstance(x, str) for x in mat)
        return False

    fit_type = settings['algorithm']
    print('Parsing \'spin_system\' and \'exchange\' blocks...')
    yaml_spin, yaml_exch = yaml.get('spin_system'), yaml.get('exchange', dict())
    if yaml_spin is None:
        raise RuntimeError('No \'spin_system\' list found')
    elif not isinstance(yaml_spin, list):
        raise TypeError('Check hyphenation of \'spin_system\' list; a list was expected but a '
                        f'{type(yaml_spin)} was found')
    elif not all(isinstance(x, dict) for x in yaml_spin):
        raise ValueError('Trouble interpreting \'spin_system\' list')
    spins = [fraction_to_float(x.get('S', 0)) for x in yaml_spin]
    if any(x <= 0 for x in spins):
        raise ValueError('All spins must have \'S\' > 0')
    elif not all(s > 0 and (2 * s) % 1 == 0 for s in spins):
        raise ValueError(f'Inappropriate spin value found in {spins}')
    spin_system = SpinSystem(spins)
    g_tensor = []
    for x in yaml_spin:
        val = x.get('g', 2.0023)
        if isinstance(val, numeric) or isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            raise TypeError(f'Non-list g tensor error caused by {val}, which is a {type(val)}')
        if not check_shape(val, [1, 2, 3, 6]):
            raise ValueError(f'Uninterpretable g tensor {val}')
        g_tensor.append(val)
    d_tensor = []
    for x in yaml_spin:
        val = x.get('D', [])
        if isinstance(val, numeric) or isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            raise ValueError(f'Non-list D tensor error caused by {val}, which is a {type(val)}')
        if not check_shape(val, [0, 1, 2, 3, 5, 6]):
            raise ValueError(f'Uninterpretable D tensor {val}')
        d_tensor.append(val)
    if spin_system.nspins == 1:
        if len(yaml_exch) != 0:
            raise RuntimeError('Single spin systems should not have exchange')
        exchange_j, exchange_g, exchange_d, j_factor = [[], [], [], -2]
    else:
        pairs = int(spin_system.nspins * (spin_system.nspins - 1) / 2)
        if isinstance(yaml_exch, numeric) or isinstance(yaml_exch, (str, list)):
            yaml_exch = {'scalar': yaml_exch}
        if yaml_exch.get('isotropic') is not None and yaml_exch.get('scalar') is not None:
            raise RuntimeError('Cannot specify both \'scalar\' and \'isotropic\' exchange coupling')
        if yaml_exch.get('antisymmetric') is not None and yaml_exch.get('vector') is not None:
            raise RuntimeError('Cannot specify both \'antisymmetric\' and \'vector\' exchange coupling')
        if yaml_exch.get('anisotropic') is not None and yaml_exch.get('tensor') is not None:
            raise RuntimeError('Cannot specify both \'anisotropic\' and \'tensor\' exchange coupling')
        if yaml_exch.get('isotropic') is not None:
            yaml_exch['scalar'] = yaml_exch['isotropic']
        if yaml_exch.get('antisymmetric') is not None:
            yaml_exch['vector'] = yaml_exch['antisymmetric']
        if yaml_exch.get('anisotropic') is not None:
            yaml_exch['tensor'] = yaml_exch['anisotropic']
        exchange_j = yaml_exch.get('scalar', [])
        exchange_g = yaml_exch.get('vector', [])
        exchange_d = yaml_exch.get('tensor', [])
        j_factor = yaml_exch.get('factor', -2)
        if not isinstance(j_factor, numeric):
            raise ValueError('J factor must be numeric')
        if not isinstance(exchange_j, list):
            exchange_j = [exchange_j]
        if not check_shape(exchange_j, [0, pairs]):
            raise RuntimeError(f'Exchange lists must have 0 or {pairs} entries and numeric/variable entries but the '
                               f'scalar coupling list was {exchange_j}')
        if not isinstance(exchange_g, list):
            raise TypeError(f'List expected in vector exchange {exchange_g}')
        if check_shape(exchange_g, [3]):
            exchange_g = [exchange_g]
        if len(exchange_g) not in [0, pairs]:
            raise RuntimeError(f'Exchange lists must have 0 or {pairs} entries '
                               f'but {exchange_g} has {len(exchange_g)}')
        if not all(check_shape(x, [0, 3]) for x in exchange_g):
            raise ValueError(f'Uninterpretable entry in vector exchange list {exchange_g}')
        if not isinstance(exchange_d, list):
            raise TypeError(f'List expected in tensor exchange {exchange_d}')
        if check_shape(exchange_d, [1, 2, 3, 5, 6]):
            exchange_d = [exchange_d]
        if len(exchange_d) not in [0, pairs]:
            raise RuntimeError(f'Exchange lists must have 0 or {pairs} entries; check tensor list')
        if not all(check_shape(x, [0, 1, 2, 3, 5, 6]) for x in exchange_d):
            raise ValueError(f'Uninterpretable entry in tensor exchange list {exchange_d}')
    variable_system = VariableSystem(g_tensor, d_tensor, exchange_j, exchange_g, exchange_d)
    if len(variable_system) > 0:
        if run_type == 'simulation':
            raise RuntimeError('Floating variables only allowed in fitting')
        print(f'Located variables: {", ".join(variable_system.names)}')
        bounds = {name: yaml['variables'][name][:2] for name in variable_system.names}
        if fit_type == 'hj':
            start_point = [yaml['variables'][name][2] for name in variable_system.names]
        else:
            start_point = [(x + y) / 2 for x, y in [bounds[name] for name in variable_system.names]]
        if fit_type == 'grid':
            d = []
            for name in variable_system.names:
                if len(yaml['variables'][name]) == 3:
                    d.append(yaml['variables'][name][2])
                else:
                    d.append(settings['divisions'])
            settings['divisions'] = d
        variable_system.set_values(start_point)
        variable_system.set_bounds(bounds)
        spin_system.set_parameters(*variable_system.resolve(start_point), j_factor=j_factor)
    else:
        spin_system.set_parameters(g_tensor, d_tensor, exchange_j, exchange_g, exchange_d, j_factor)
    return spin_system, variable_system, settings


def read_integration(yaml, spin_system) -> IntegrationGrid:
    print('Parsing \'integration\' block...')
    yaml_integ = yaml.get('integration', {})
    if not isinstance(yaml_integ, dict):
        raise TypeError(f'Improperly formatted \'integration\' block; received {type(yaml_integ)} instead of dict. '
                        'Remove any hyphenation')
    method = lower(yaml_integ.get('method', 'lebedev'))
    if method not in ['lebedev', 'gaussian', 'custom']:
        raise ValueError(f'Unrecognized or unimplemented integration method {method}')
    precision = lower(yaml_integ.get('precision', 'auto'))
    if precision == 'auto':
        precision = {'lebedev': 19, 'gaussian': 5, 'custom': 0}[method]
    if method == 'lebedev':
        if not isinstance(precision, int):
            raise TypeError(f'Lebedev precision must be an integer but {precision} is a {type(precision)}')
    if method == 'gaussian':
        if not isinstance(precision, (int, list)):
            raise TypeError(f'Gaussian precision must be an integer or list but {precision} is a {type(precision)}')
    domain = lower(yaml_integ.get('domain'))
    if method == 'custom':
        if domain is None:
            raise ValueError('A custom grid expects a filename in \'domain\' field')
        if not isinstance(domain, str):
            raise TypeError('A custom grid expects a string in \'domain\' field as a filename')
        print(f'- Reading user-specified grid in \'{domain}\'')
    else:
        if domain is None:
            domain = 'auto'
        if domain not in ['sphere', 'hemi', 'hemisphere', 'octant', 'auto']:
            raise ValueError('Unrecognized integration domain')
    if domain == 'hemi':
        domain = 'hemisphere'
    if domain == 'auto':
        domain = {'centrosymmetric': 'hemisphere', 'orthorhombic': 'octant',
                  'axial': 'octant', 'isotropic': 'octant'}[spin_system.symmetry]
    if spin_system.symmetry == 'centrosymmetric' and domain == 'octant':
        print('WARNING: Octant domain likely inappropriate for this spin')
        print('  Hamiltonian! Euler angles and/or antisymmetric exchange remove')
        print('  orthorhombic symmetry such that `domain: hemisphere` should be used!')
    return IntegrationGrid(method, precision, domain)


def read_fit_settings(yaml, run_type) -> dict:
    settings = {'type': run_type}
    if run_type == 'fit':
        print('Parsing \'fitting\' block...')
        yaml_fit = yaml.get('fitting', dict())
        if not isinstance(yaml_fit, dict):
            raise TypeError(f'Uninterpretable \'fitting\' block; expected dict but got {type(yaml_fit)}')
        cutoff = yaml_fit.get('svd')
        c_degree = yaml_fit.get('dimension')
        if cutoff is None and c_degree is None:
            cutoff = 1e-4
        elif cutoff is None and c_degree is not None:
            if not isinstance(c_degree, int):
                raise TypeError(f'\'dimension\' must be an integer but {c_degree} is a {type(c_degree)}')
            if c_degree < 1:
                raise ValueError(f'\'dimension\' must be positive but {c_degree} received')
            cutoff = -c_degree
        elif cutoff is not None and c_degree is None:
            if not isinstance(cutoff, numeric):
                raise TypeError(f'\'svd\' cutoff must be numeric but {cutoff} is a {type(cutoff)}')
            if cutoff < 0:
                raise ValueError(f'\'svd\' cutoff must be non-negative but {cutoff} received')
            cutoff = abs(cutoff)
        else:  # both are specified
            raise RuntimeError('Cannot specify both \'svd\' cutoff and \'dimension\' size')
        settings['algorithm'] = yaml_fit.get('algorithm', 'ps')
        settings['cutoff'] = cutoff
        settings['parallel'] = yaml_fit.get('parallel', 1)
        settings['iterations'] = yaml_fit.get('iterations', 100)
        settings['covariance'] = yaml_fit.get('covariance', None)
        if settings['algorithm'] == 'ps':  # Particle Swarm is the default
            settings['cognitive'] = yaml_fit.get('cognitive', 2.5)
            settings['social'] = yaml_fit.get('social', 1.3)
            settings['inertial'] = yaml_fit.get('inertial', 0.4)
            settings['particles'] = yaml_fit.get('particles', 20)
            settings['history'] = yaml_fit.get('history', False)
            if not isinstance(settings['history'], bool):
                raise TypeError(f'\'history\' must be boolean not {type(settings["history"])}')
        elif settings['algorithm'] == 'hj':  # Hooke & Jeeves
            settings['rho'] = yaml_fit.get('rho', 0.7)
            settings['eps'] = yaml_fit.get('eps', 1.0e-8)
        elif settings['algorithm'] == 'grid':  # Grid Search
            settings['divisions'] = yaml_fit.get('divisions', 5)
            settings['start'] = yaml_fit.get('start', 0)
    else:  # 'simulation'
        settings['algorithm'] = 'sim'
        print('Skipping any \'fitting\' block... (simulation)')
    return settings


def read_data_files(yaml, run_type, integration_grid) -> DataCollection:
    print('Parsing \'data\' block...')
    yaml_data = yaml.get('data')
    dim = 3 if integration_grid.domain == 'octant' else 9  # polarization dimension
    if yaml_data is None:
        raise RuntimeError('YAML file is missing \'data\' section')
    if isinstance(yaml_data, dict):
        raise TypeError('Datafiles must be provided as a list not a dict; double check hyphenation (-) is used')
    if not isinstance(yaml_data, list):
        raise TypeError(f'Datafiles must be provided as a list not a {type(yaml_data)}')
    if not all(isinstance(x, dict) for x in yaml_data):
        raise TypeError('Datafiles must be provided as a list of dict entries')
    datasets = []
    for i in range(len(yaml_data)):
        infile = yaml_data[i].get('file')
        if infile is None:
            raise RuntimeError(f'File name missing for data file {i + 1}')
        if not isinstance(infile, str):
            raise RuntimeError(f'Non-string \'file\' name {infile} (type {type(infile)})')
        outfile = yaml_data[i].get('out')
        if outfile is None:
            if infile[-4:].lower() == '.csv':
                outfile = infile[:-4] + '.out' + infile[-4:]
            else:
                outfile = infile + '.out.csv'
        else:
            if not isinstance(outfile, str):
                raise RuntimeError(f'Non-string \'out\' name {outfile} (type {type(outfile)})')
        print(f'- Reading document {i + 1} \'{infile}\'')
        dataset = Dataset(infile, outfile)
        if run_type == 'fit':
            number_of_transitions = dataset.data.shape[0]
            if number_of_transitions == 0:
                raise RuntimeError(f'No data found in file {infile}')
            constraints = yaml_data[i].get('constraints', [None] * number_of_transitions)
            if not isinstance(constraints, list):
                raise TypeError('Polarization constraints must be supplied as an array but '
                                f'{constraints} is of type {type(constraints)}')
            if len(constraints) != number_of_transitions:
                raise RuntimeError('Number of polarization constraints differs from number of transitions '
                                   f'in file {infile}')
            for j in range(number_of_transitions):
                if constraints[j] is None:
                    continue
                elif isinstance(constraints[j], str) and lower(constraints[j]) == 'none':
                    constraints[j] = None
                elif isinstance(constraints[j], list):
                    constraints[j] = np.array(constraints[j])
                    if constraints[j].dtype == np.dtype('O'):
                        raise TypeError(f'Confusing {ordinal(j + 1)} constraint {constraints[j]}')
                    if len(constraints[j].shape) == 1:  # format [x, y, z] -> [[x, y, z]]
                        constraints[j] = constraints[j][np.newaxis, :]
                    if len(constraints[j].shape) > 2:
                        raise ValueError(f'Confusing {ordinal(j + 1)} constraint {constraints[j]}')
                    if constraints[j].shape[1] != dim:
                        raise ValueError(f'Constraint supplied as a {constraints[j].shape[1]}-entry array but {dim} '
                                         f'entries are expected with a(n) {integration_grid.domain} integration grid')
                else:
                    raise TypeError(f'Polarization constraints {constraints[j]} should be \'None\' or a list')
            dataset.constraints = constraints
            b_term = yaml_data[i].get('b_term', [None] * number_of_transitions)
            if not isinstance(b_term, list):
                b_term = [b_term]
            if len(b_term) != number_of_transitions:
                raise RuntimeError('Number of polarization B-term values differs from number of transitions '
                                   f'in file {infile}')
            for j in range(number_of_transitions):
                if b_term[j] is None:
                    continue
                elif isinstance(b_term[j], numeric):
                    if b_term[j] == 0:
                        b_term[j] = None
                elif isinstance(b_term[j], str):
                    if lower(b_term[j]) == 'none':
                        b_term[j] = None
                    elif lower(b_term[j]) == 'fit':
                        b_term[j] = 'fit'
                    else:
                        raise ValueError(f'Uninterpretable B-term specification: {b_term[j]}')
                else:
                    raise TypeError(f'Polarization B-term {b_term[j]} should be \'None\' or \'fit\' or a number')
            dataset.b_term = b_term
        else:  # 'simulation'
            polarizations = yaml_data[i].get('polarizations')
            if polarizations is None:
                raise RuntimeError('Simulations require specification of polarizations')
            if not isinstance(polarizations, list):
                raise TypeError(f'Polarizations should be an array but are {type(polarizations)}')
            if len(polarizations) == 0:
                raise ValueError('Simulations require specification of polarizations')
            polarizations = np.array(polarizations)
            if len(polarizations.shape) == 1:
                if polarizations.dtype == np.dtype('O'):
                    raise TypeError(f'Improper polarization list {polarizations}')
                polarizations = polarizations[np.newaxis, :]
            if len(polarizations.shape) > 2:
                raise ValueError(f'Polarization array deeper than expected for a simulation')
            number_of_transitions = polarizations.shape[0]
            if polarizations.shape[1] != dim:
                raise ValueError(f'Polarizations supplied as a {polarizations.shape[1]}-entry array but {dim} '
                                 f'entries are expected with a(n) {integration_grid.domain} integration grid')
            dataset.polarizations = polarizations
            b_term = yaml_data[i].get('b_term', [None] * number_of_transitions)
            if not isinstance(b_term, list):
                raise TypeError('Polarization B-term value must be supplied as an array but '
                                f'{b_term} is of type {type(constraints)}')
            if len(b_term) != number_of_transitions:
                raise RuntimeError('Number of polarization B-term values differs from number of transitions '
                                   f'in file {infile}')
            for j in range(number_of_transitions):
                if b_term[j] is None:
                    continue
                elif isinstance(b_term[j], numeric):
                    continue
                elif isinstance(b_term[j], str):
                    if lower(b_term[j]) == 'none':
                        b_term[j] = None
                    else:
                        raise ValueError(f'Uninterpretable B-term specification: {b_term[j]}')
                else:
                    raise TypeError(f'Polarization B-term {b_term[j]} should be \'None\' or or a number')
            dataset.b_term = b_term
        center_list = yaml_data[i].get('centers', [1] * number_of_transitions)
        if isinstance(center_list, int):
            center_list = [center_list]
        if not isinstance(center_list, list):
            raise TypeError(f'Centers are expected as a list but {center_list} is a {type(center_list)}')
        if len(center_list) != number_of_transitions:
            raise RuntimeError(f'Mismatch of {number_of_transitions} transitions found but only '
                               f'{len(center_list)} centers specified')
        dataset.centers = [x - 1 for x in center_list]  # Python counts from 0 but users will count from 1
        datasets.append(dataset)
    return DataCollection(datasets)


def read_yaml(filename: str) -> (SpinSystem, VariableSystem, IntegrationGrid, DataCollection, dict):
    with open(filename, 'r') as f:
        yaml = safe_load(f)

    run_type = {'sim': 'simulation', 'simulation': 'simulation',
                'fit': 'fit', 'fitting': 'fit'}.get(lower(yaml.get('type')))
    if run_type is None:
        raise ValueError('Run type not specified or unrecognized; try \'fit\' or \'simulation\'')

    settings = read_fit_settings(yaml, run_type)
    spin_system, variable_system, settings = read_spin_system(yaml, run_type, settings)
    integration_grid = read_integration(yaml, spin_system)
    data_collection = read_data_files(yaml, run_type, integration_grid)

    print('\nSummary of job:')
    print('- Run type:', settings['type'])
    print('- Data file(s):')
    for i, dataset in enumerate(data_collection.datasets):
        print(f'  {i + 1}) {dataset.filename} with {dataset.tempfield.shape[0]} point(s) and data for '
              f'{dataset.data.shape[0]} transition(s)')
        if dataset.header > 0:
            print(f'    - Number of header lines skipped: {dataset.header}')
        print(f'    - Output file \'{dataset.output}\'')
        for j in range(len(dataset.centers)):
            if settings['type'] == 'simulation':
                if not dataset.b_term[j]:
                    print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1} polarized as',
                          str(dataset.polarizations[j]))
                else:
                    print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1}')
                    print(f'      - Polarized as', str(dataset.polarizations[j]))
                    print(f'      - Exhibiting B-term intensity of', str(dataset.b_term[j]), 'T^-1')
            else:  # 'fit'
                if dataset.constraints[j] is None:
                    print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1} unconstrained')
                else:
                    if dataset.constraints[j].shape[0] == 1:
                        print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1} constrained',
                              'along vector of', str(dataset.constraints[j][0]))
                    elif dataset.constraints[j].shape[0] == 2:
                        print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1} constrained',
                              'within plane of', str(dataset.constraints[j][0]), 'and', str(dataset.constraints[j][1]))
                    else:
                        print(f'    - Transition {j + 1} on center {dataset.centers[j] + 1} constrained',
                              'within hyperplane of', ', '.join(str(dataset.constraints[j][k])
                                                                for k in range(dataset.constraints[j].shape[1])))
                if dataset.b_term[j] == 'fit':
                    print('      - Temperature-independent B-term intensity will be fitted')
                elif dataset.b_term[j]:
                    print('      - Temperature-independent B-term intensity set to', str(dataset.b_term[j]), 'T^-1')
    print('- Numerical integration:')
    print(f'  - Method: {integration_grid.method}')
    if integration_grid.method != 'custom':
        print(f'  - Precision: {integration_grid.precision} ({integration_grid.size} points)')
        print(f'  - Domain: {integration_grid.domain}')
    else:
        print(f'  - Domain: user-specified from file {integration_grid.source}')
    print('- Spin Hamiltonian')
    for i in range(spin_system.nspins):
        print(f'  - {ordinal(i + 1)} spin, S = {spin_to_fraction(spin_system.spins[i])}')
        print(f'      g = [{", ".join(str(w) for w in variable_system.inputs["g_tensor"][i])}]')
        print(f'      D = [{", ".join(str(w) for w in variable_system.inputs["d_tensor"][i])}]')
    if spin_system.nspins > 1:
        if len(variable_system.exchange_j) == 0:
            print('  - Scalar (isotropic) J exchange coupling: None')
        else:
            if spin_system.j_factor >= 0:
                print(f'  - Using a +{spin_system.j_factor}J term')
            else:
                print(f'  - Using a {spin_system.j_factor}J term')
            print('  - Scalar (isotropic) J exchange coupling:')
            for i in range(len(spin_system.jindices)):
                print(f'      J{spin_system.jindices[i][0] + 1}{spin_system.jindices[i][1] + 1} =',
                      str(variable_system.inputs['exchange_j'][i]))
        if len(variable_system.exchange_g) == 0:
            print('  - Vector (antisymmetric) G exchange coupling: None')
        else:
            print('  - Vector (antisymmetric) G exchange coupling:')
            for i in range(len(spin_system.jindices)):
                print(f'      G{spin_system.jindices[i][0] + 1}{spin_system.jindices[i][1] + 1} =',
                      str(variable_system.inputs['exchange_g'][i]))
        if len(variable_system.exchange_g) == 0:
            print('  - Tensor (anisotropic) D exchange coupling: None')
        else:
            print('  - Tensor (anisotropic) D exchange coupling:')
            for i in range(len(spin_system.jindices)):
                print(f'      D{spin_system.jindices[i][0] + 1}{spin_system.jindices[i][1] + 1} =',
                      str(variable_system.inputs['exchange_d'][i]))
    if settings['type'] == 'fit':
        print('- Fitting settings:')
        if len(variable_system) > 0 and settings['algorithm'] == 'ps':
            print('  - Fitting algorithm for variables: particle swarm')
            print(f'  - Found {len(variable_system)} variable(s) to be fit:')
            for i in range(len(variable_system)):
                print(f'      {variable_system.names[i]} in [{", ".join(str(x) for x in variable_system.bounds[i])}]')
            print('  - Cognitive/personal (c1) parameter:', settings['cognitive'])
            print('  - Social/group (c2) parameter:', settings['social'])
            print('  - Inertial (w) parameter:', settings['inertial'])
            print('  -', settings['particles'], 'particles')
            print('  -', settings['iterations'], 'iterations')
            if settings['parallel'] > 1:
                print('  -', settings['parallel'], 'parallel processes')
            else:
                print('  - Serial execution')
            if settings['history']:
                settings['history'] = filename[:-5] if filename[-5:] == '.yaml' else filename
                print('  - Particle history will be written to\n'
                      f'    - Particle costs in {settings["history"]}.cost.csv\n'
                      f'    - Particle positions in {settings["history"]}.position.csv\n'
                      f'    - Particle velocities in {settings["history"]}.velocity.csv')
        elif len(variable_system) > 0 and settings['algorithm'] == 'hj':
            print('  - Fitting algorithm for variables: Hooke & Jeeves')
            print(f'  - Found {len(variable_system)} variable(s) to be fit:')
            for i in range(len(variable_system)):
                print(f'      {variable_system.names[i]} in [{", ".join(str(x) for x in variable_system.bounds[i])}]')
            print('  - Scaling parameter (rho):', settings['rho'])
            print('  - Convergence criterion (eps):', settings['eps'])
            print('  - Maximum iterations:', settings['iterations'])
            if settings['parallel'] > 1:
                print('  -', settings['parallel'], 'parallel processes')
            else:
                print('  - Serial execution')
        elif len(variable_system) > 0 and settings['algorithm'] == 'grid':
            print('  - Fitting algorithm for variables: Grid Search')
            print(f'  - Found {len(variable_system)} variable(s) to be fit:')
            for i in range(len(variable_system)):
                span = np.linspace(variable_system.bounds[i][0], variable_system.bounds[i][1],
                                   settings['divisions'][i])
                print(f'      {variable_system.names[i]} in [{", ".join(str(x) for x in span)}]')
            if settings['parallel'] > 1:
                print('  -', settings['parallel'], 'parallel processes')
            else:
                print('  - Serial execution')
        else:
            print('  - No variables require particle swarm fitting')
        if settings['cutoff'] >= 0:
            print('  - Dimensionality of C matrix by SVD, cutoff', str(settings['cutoff']))
        else:
            print('  - Dimensionality of C matrix set to', str(abs(settings['cutoff'])))
    return spin_system, variable_system, integration_grid, data_collection, settings


def report(fit_system: FitSystem):
    print('\nSummary of fit:')
    if len(fit_system.variable_system) == 0:
        print('- No fitted variables')
    else:
        print('- Fitted variables:')
        for n in range(len(fit_system.variable_system)):
            print('  - Refined', fit_system.variable_system.names[n], 'to',
                  str(fit_system.variable_system.values[n]))
        print('- Spin Hamiltonian using fitted variables:')
        g_tensor, d_tensor, exchange_j, exchange_g, exchange_d = fit_system.variable_system.resolve()
        for i in range(fit_system.spin_system.nspins):
            print(f'  - {ordinal(i + 1)} spin, S = {spin_to_fraction(fit_system.spin_system.spins[i])}')
            print(f'      g = {g_tensor[i]}')
            print(f'      D = {d_tensor[i]}')
        if fit_system.spin_system.nspins > 1:
            j_indices = fit_system.spin_system.jindices
            if len(exchange_j) == 0:
                print('  - Scalar (isotropic) J exchange coupling: None')
            else:
                if fit_system.spin_system.j_factor >= 0:
                    print(f'  - Using a +{fit_system.spin_system.j_factor}J term')
                else:
                    print(f'  - Using a {fit_system.spin_system.j_factor}J term')
                print('  - Scalar (isotropic) J exchange coupling:')
                for i in range(len(j_indices)):
                    print(f'      J{j_indices[i][0] + 1}{j_indices[i][1] + 1} =', str(exchange_j[i]))
            if len(exchange_g) == 0:
                print('  - Vector (antisymmetric) G exchange coupling: None')
            else:
                print('  - Vector (antisymmetric) G exchange coupling:')
                for i in range(len(j_indices)):
                    print(f'      G{j_indices[i][0] + 1}{j_indices[i][1] + 1} =', str(exchange_g[i]))
            if len(exchange_d) == 0:
                print('  - Tensor (anisotropic) D exchange coupling: None')
            else:
                print('  - Tensor (anisotropic) D exchange coupling:')
                for i in range(len(j_indices)):
                    print(f'      D{j_indices[i][0] + 1}{j_indices[i][1] + 1} =', str(exchange_d[i]))
    print('- Weighted residual sum of squares:', sum(fit_system.cost))
    if len(fit_system.cost) > 1:  # if we have multiple data sets, print individual costs
        for n in range(len(fit_system.cost)):
            print('  - Dataset', str(n + 1), 'with WRSS =', str(fit_system.cost[n]))

    print('\nStatistical Analysis:')
    print('- Estimating residual Jacobian matrix by numerical differentiation')
    print('- Using reciprocal variance as weighting scheme')
    n_vars = len(fit_system.variable_system)
    covariance = fit_system.covariance_matrix()
    if n_vars == 0:
        print('- No fitted variables')
    else:
        print('- Fitted variables:')
        print('  - (Co)variances, weighted:')
        print_matrix_eq('cov(', [[name] for name in fit_system.variable_system.names], ') = ',
                        covariance[:n_vars, :n_vars])
        print('  - Correlation coefficients, weighted:')
        corr = np.diag(np.diag(covariance[:n_vars, :n_vars]) ** -0.5)
        corr = corr @ covariance[:n_vars, :n_vars] @ corr
        print_matrix_eq('corr(', [[name] for name in fit_system.variable_system.names], ') = ', corr)
        print('  - Standard error of the fit, weighted:')
        for v in range(n_vars):
            error = covariance[v, v]
            if not np.isnan(error):
                error = error ** 0.5
            print(f'    - Variable {fit_system.variable_system.names[v]} =',
                  f'{fit_system.variable_system.values[v]} +/- {error}')
    all_transforms = [np.eye(n_vars)]
    p_names, count = [], 0
    p_info = fit_system.polarization_info()
    for d, dataset in enumerate(fit_system.data_collection):
        print(f'- Polarization info for dataset {d + 1} ({dataset.filename})')
        pols = p_info[d]
        for t in range(len(pols)):
            if len(pols) > 1:
                print(f'  - {ordinal(t + 1)} transition')
            cartesian = [['Myz'], ['Mzx'], ['Mxy']]
            params, transform = pols[t]
            all_transforms.append(sorting_transform(transform))
            sub_covariance = covariance[(n_vars + count):(n_vars + count + len(transform)),
                                        (n_vars + count):(n_vars + count + len(transform))]
            sub_covariance = all_transforms[-1] @ sub_covariance @ all_transforms[-1].T
            # reduced @ transform = xdata and reduced @ params = ydata
            # if reduced = xdata @ transform.T then xdata @ transform.T @ params = ydata
            # so transform.T @ params = [Myz, Mzx, Mxy(, B)]
            # or equivalently params = transform @ [Myz, Mzx, Mxy(, B)]
            if dataset.b_term[t]:
                if isinstance(dataset.b_term[t], numeric):
                    print(f'  - Temperature-independent B term of {dataset.b_term[t]} T^-1')
                if dataset.b_term[t] == 'fit':
                    cartesian.append(['B'])
            names = []
            for p in params:
                names.append('M' + str(count + 1))
                count += 1
            p_names.extend(names)
            corr = np.diag(np.diag(sub_covariance) ** -0.5)
            corr = corr @ sub_covariance @ corr
            print_matrix_eq(all_transforms[-1] @ transform, ' * ', cartesian, ' = ',
                            [[name] for name in names], ' = ',
                            all_transforms[-1] @ params[:, np.newaxis], ' +/- ',
                            np.diag(sub_covariance)[:, np.newaxis] ** 0.5)
            print_matrix_eq('  corr(', [[name] for name in names], ') = ', corr)
    if fit_system.settings['covariance'] is not None:
        covariance_filename = fit_system.settings['covariance']
        all_labels = list(fit_system.variable_system.names) + list(p_names)
        with open(covariance_filename, mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(all_labels)
            for r in covariance:
                writer.writerow(r)


if __name__ == '__main__':
    print('MCD Simulation and Fitting with Particle Swarm Optimization\n'
          'Version 0.7, Wesley J. Transue\n\n'
          'Citation: Transue, W. J.; Solomon, E. I. Title. Journal 2022,\n'
          'Vol (Issue), Pages. DOI: 10.XXXX/YYYYYY\n')
    if len(sys.argv) != 2:
        raise RuntimeError('Provide one YAML input file')

    start_time = time()
    spin_system, variable_system, integration_grid, data_collection, settings = read_yaml(sys.argv[1])

    print('\nBeginning Job...', flush=True)
    simulations = []
    if settings['type'] == 'fit':
        fit = FitSystem(spin_system, variable_system, integration_grid, data_collection, settings)
        if fit.variable_system:  # if there are variables to be fit, deploy particle swarm
            if settings['algorithm'] == 'ps':
                fit.particle_swarm_fit()
                print('Particle swarm variable optimization finished')
            elif settings['algorithm'] == 'hj':
                fit.hooke_and_jeeves_fit()
                print('Hooke and Jeeves optimization finished')
            elif settings['algorithm'] == 'grid':
                fit.grid_search_fit()
                print('Grid Search optimization finished')
        else:
            fit.fit_no_variables()
        for dataset in fit.data_collection:
            dataset.spin_system = fit.spin_system
            dataset.integration_grid = fit.integration_grid
            dataset.cutoff = settings['cutoff']
            dataset.calculate_by_fit()
        report(fit)
    else:
        for dataset in data_collection:
            dataset.spin_system = spin_system
            dataset.integration_grid = integration_grid
            dataset.cutoff = 0
            dataset.calculate_given()
    for dataset in data_collection:
        dataset.write_file()
    express_duration(start_time)
