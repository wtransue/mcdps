#
# Contains classes and subroutines relevant to construction of the `SpinSystem` object, which is central to the
# function of the VTVH MCD fitting program
#

import numpy as np
from os import environ


def euler_rotate(a, b, c) -> np.ndarray:
    """Converts zyz Euler angles (in degrees) into a rotation matrix

    :param alpha: First Euler angle in degrees
    :param beta: Second Euler angle in degrees
    :param gamma: Third Euler angle in degrees
    :return: Rotation matrix
    """
    # 0.017453292519943295 is pi / 180
    alpha, beta, gamma = 0.017453292519943295 * a, 0.017453292519943295 * b, 0.017453292519943295 * c
    c1, s1 = np.cos(alpha), np.sin(alpha)
    c2, s2 = np.cos(beta),  np.sin(beta)
    c3, s3 = np.cos(gamma), np.sin(gamma)
    return np.array([[ c1 * c2 * c3 - s1 * s3, c1 * s3 + c2 * c3 * s1, -c3 * s2],
                     [-c3 * s1 - c1 * c2 * s3, c1 * c3 - c2 * s1 * s3,  s2 * s3],
                     [ c1 * s2,                s1 * s2,                 c2]])


def get_g_mat(g_tensor) -> np.ndarray:
    """Constructs a 3x3 g matrix from a 6-membered list containing Euler angle information

    :param g_tensor: A six-membered list [gx, gy, gz, alpha, beta, gamma]
    :return: g tensor (3x3 matrix)
    """
    if len(g_tensor) != 6:
        raise ValueError(f'Uninterpretable g tensor {g_tensor}')
    gx, gy, gz, alpha, beta, gamma = g_tensor
    rot = euler_rotate(alpha, beta, gamma)
    g = [[gx, 0, 0], [0, gy, 0], [0, 0, gz]]
    return rot @ g @ rot.T


def get_d_mat(d_tensor):
    """
    get_d_mat: construct 3x3 ZFS matrix from a 5- or 6-membered list
    5: [D, E/D, alpha, beta, gamma] using zyz Euler angles
    6: [Dxx, Dyy, Dzz, alpha, beta, gamma] using zyz Euler angles
    Automatically corrects [Dxx, Dyy, Dzz] to be traceless
    """
    if len(d_tensor) == 5:
        d, eod, alpha, beta, gamma = d_tensor
        dmat = np.diag(d * np.array([-1 / 3 + eod, -1 / 3 - eod, 2 / 3]))
    elif len(d_tensor) == 6:  # len(d_tensor) == 6
        alpha, beta, gamma = d_tensor[3:]
        dmat = np.array(d_tensor[:3])
        dmat = np.diag(dmat - sum(dmat) / 3)
    else:
        raise ValueError(f'Uninterpretable D tensor {d_tensor}')
    rot = euler_rotate(alpha, beta, gamma)
    return rot @ dmat @ rot.T


class Hamiltonian:
    """Separately stores field-independent and field-dependent pieces of Hamiltonian matrix for easy/rapid recalculation
    at each point in the integration grid"""

    def __init__(self, field_independent, field_dependent):
        self.indep = np.array(field_independent)
        # because of the way np.dot (@) works, we must transpose the field-dependent portion as (1, 0, 2) for `at_field`
        self.dep = np.array(field_dependent).transpose((1, 0, 2))

    def at_field(self, field) -> np.ndarray:
        """Constructs a matrix representation of the spin Hamiltonian at the provided field

        :param field: A list formatted as [Bx, By, Bz]
        :return: ndarray (matrix) of the spin Hamiltonian
        """
        return self.indep + field @ self.dep


class SpinSystem:
    """Stores info on magnetic parameters and variables; constructs `Hamiltonian` objects when supplied with variable
    values; reports spin expectation values for a given states"""

    def __init__(self, spinlist):
        def spinx(S) -> np.ndarray:
            """Returns an S_X operator matrix representation for a spin-S system"""
            return np.array([[(int(m == n + 1) + int(m + 1 == n)) * np.sqrt(S * (S + 1) - m * n) / 2
                              for n in np.linspace(S, -S, int(2 * S + 1))] for m in np.linspace(S, -S, int(2 * S + 1))])

        def spiny(S) -> np.ndarray:
            """Returns an S_Y operator matrix representation for a spin-S system"""
            return np.array([[(int(m == n + 1) - int(m + 1 == n)) * np.sqrt(S * (S + 1) - m * n) / 2j
                              for n in np.linspace(S, -S, int(2 * S + 1))] for m in np.linspace(S, -S, int(2 * S + 1))])

        def spinz(S) -> np.ndarray:
            """Returns an S_Z operator matrix representation for a spin-S system"""
            return np.array([[int(m == n) * n
                              for n in np.linspace(S, -S, int(2 * S + 1))] for m in np.linspace(S, -S, int(2 * S + 1))])

        def ident(S) -> np.ndarray:
            """Returns the identity operator matrix for a spin-S system"""
            return np.identity(round(2 * S + 1))

        def tensor_product(list_of_mats) -> np.ndarray:
            """A flexible Kronecker product function that operates over (ndarray) matrices

            :param list_of_mats: list of (list/ndarray) matrices to be used in the Kronecker product
            :return: ndarray of the tensor/Kronecker product
            """
            if len(list_of_mats) == 1:
                return list_of_mats[0]
            else:
                return np.kron(list_of_mats[0], tensor_product(list_of_mats[1:]))

        self.spins = list(spinlist)
        self.nspins = len(spinlist)
        self.operators = np.array([[tensor_product([spinx(spinlist[v]) if u == v else ident(spinlist[v])
                                                    for v in range(self.nspins)]),
                                    tensor_product([spiny(spinlist[v]) if u == v else ident(spinlist[v])
                                                    for v in range(self.nspins)]),
                                    tensor_product([spinz(spinlist[v]) if u == v else ident(spinlist[v])
                                                    for v in range(self.nspins)])] for u in range(self.nspins)])
        self.jindices = [[i, j] for i in range(self.nspins) for j in range(i+1, self.nspins)]
        self._zeros = int(np.prod(2 * np.array(self.spins) + 1))
        self._zeros = np.zeros((self._zeros, self._zeros)).astype('complex128')
        self._symmetry = None  # None unset, 0 centro, 1 ortho, 2 axial, 3 iso
        self.g_tensor = None
        self.d_tensor = None
        self.exchange_j = None
        self.exchange_g = None
        self.exchange_d = None
        self.j_factor = None

    def set_parameters(self, g_tensor, d_tensor, exchange_j=None, exchange_g=None, exchange_d=None, j_factor=-2):
        """Reassigns the spin Hamiltonian parameters"""
        if exchange_j is None:
            exchange_j = []
        if exchange_g is None:
            exchange_g = []
        if exchange_d is None:
            exchange_d = []
        self.g_tensor = [[g] if isinstance(g, (int, float)) else g for g in g_tensor]
        self.d_tensor = [[d] if isinstance(d, (int, float)) else d for d in d_tensor]
        self.exchange_j = exchange_j
        self.exchange_g = exchange_g
        self.exchange_d = [[d] if isinstance(d, (int, float)) else d for d in exchange_d]
        self.j_factor = j_factor

    @property
    def symmetry(self):
        """Reports the symmetry of the spin Hamiltonian based on the (shapes of the) tensors involved"""
        symm_g = max([1, *[len(x) for x in self.g_tensor]])
        symm_g = {6: 0, 3: 1, 2: 2, 1: 3}[symm_g]
        symm_d = max([0, *[len(x) for x in self.d_tensor]])
        symm_d = {6: 0, 5: 0, 3: 1, 2: 1, 1: 2, 0: 3}[symm_d]
        symm_ex_j = 3  # always isotropic no matter exchange_j
        symm_ex_g = max([0, *[len(x) for x in self.exchange_g]])
        symm_ex_g = {3: 0, 0: 3}[symm_ex_g]
        symm_ex_d = max([0, *[len(x) for x in self.exchange_d]])
        symm_ex_d = {6: 0, 5: 0, 3: 1, 2: 1, 1: 2, 0: 3}[symm_ex_d]
        symmetry = min(symm_g, symm_d, symm_ex_j, symm_ex_g, symm_ex_d)
        return ['centrosymmetric', 'orthorhombic', 'axial', 'isotropic'][symmetry]

    def ham(self, g_tensor=None, d_tensor=None, exchange_j=None, exchange_g=None, exchange_d=None) -> Hamiltonian:
        """Constructs a Hamiltonian object from the spin Hamiltonian parameters"""
        def to_cartesian(vec) -> np.ndarray:
            """Converts [r, theta, phi] to [x, y, z]"""
            vec_degrees = vec * np.pi / 180
            st = np.sin(vec_degrees[1])
            return vec[0] * np.array([np.cos(vec_degrees[2]) * st, np.sin(vec_degrees[2]) * st, np.cos(vec_degrees[1])])

        g_tensor = self.g_tensor if g_tensor is None else g_tensor
        d_tensor = self.d_tensor if d_tensor is None else d_tensor
        exchange_j = self.exchange_j if exchange_j is None else exchange_j
        exchange_g = self.exchange_g if exchange_g is None else exchange_g
        exchange_d = self.exchange_d if exchange_d is None else exchange_d
        ham0 = np.array(self._zeros)
        ham1 = np.array([self._zeros, self._zeros, self._zeros])

        # constructing Zeeman perturbation (field dependent)
        for i in range(self.nspins):
            this = g_tensor[i]
            if len(this) == 1:  # isotropic g; e.g. [2.0] -> [2.0, 2.0, 2.0]
                contrib = this[0] * self.operators[i]
            elif len(g_tensor[i]) == 2:  # axial g; e.g. [2.0, 2.1] -> [2.0, 2.0, 2.1]
                contrib = np.einsum('a,abc->abc', [this[0], this[0], this[1]], self.operators[i])
            elif len(g_tensor[i]) == 3:  # rhombic g; e.g. [2.0, 2.1, 2.2] -> [2.0, 2.1, 2.2]
                contrib = np.einsum('a,abc->abc', this, self.operators[i])
            elif len(g_tensor[i]) == 6:  # rhombic with rotation by Euler angles as [gx,gy,gz,alpha,beta,gamma]
                g_matrix = get_g_mat(this)
                contrib = np.einsum('ab,bcd', g_matrix, self.operators[i])
            else:  # improperly formatted
                raise ValueError(f'Uninterpretable g tensor {this}')
            ham1 += 0.4668644735835207 * contrib  # 0.4668644735835207 is Bohr magneton in cm-1/Tesla
        # constructing quadratic zero-field splitting (field independent)
        for i in range(self.nspins):
            this = d_tensor[i]
            if len(this) == 0:  # no zero field splitting (isotropic); i.e. [] -> [0,0,0]
                continue
            elif len(this) == 1:  # axial D without rotation; e.g. [3.0] = [-1.0,-1.0,2.0]
                this = this[0] * np.array([-1 / 3, -1 / 3, 2 / 3])
                contrib = np.einsum('uaw,uwb->uab', self.operators[i], self.operators[i])
                contrib = np.einsum('u,uvw', this, contrib)
            elif len(this) == 2:  # rhombic D w/o rotation [D,E/D]; e.g. [3.0,0.333333] = [-2.0,0.0,2.0]
                this = this[0] * np.array([-1 / 3 + this[1], -1 / 3 - this[1], 2 / 3])
                contrib = np.einsum('uaw,uwb->uab', self.operators[i], self.operators[i])
                contrib = np.einsum('u,uvw', this, contrib)
            elif len(this) == 3:  # rhombic D, full diagonal specified; e.g. [1,0,-1] = [1,0,-1]
                this = np.array(this) - sum(this)/3  # make traceless
                contrib = np.einsum('uaw,uwb->uab', self.operators[i], self.operators[i])
                contrib = np.einsum('u,uvw', this, contrib)
            elif len(this) == 5 or len(d_tensor[i]) == 6:  # rhombic + Euler angles [D,E/D,alpha,beta,gamma]
                this = get_d_mat(this)
                contrib = np.einsum('uaw,uv,vwb', self.operators[i], this, self.operators[i])
            else:  # improperly formatted
                raise ValueError(f'Uninterpretable D tensor {this}')
            ham0 += contrib
        # scalar (isotropic) exchange (field independent)
        if len(exchange_j) > 0:
            for i in range(len(self.jindices)):
                j, k = self.jindices[i]
                contrib = np.einsum('ijk,ikl', self.operators[j], self.operators[k])
                ham0 += self.j_factor * exchange_j[i] * contrib
        # vector (antisymmetric) exchange (field independent)
        if len(exchange_g) > 0:
            for i in range(len(self.jindices)):
                if len(exchange_g[i]) == 0:
                    continue
                if len(exchange_g[i]) != 3:
                    raise ValueError(f'Uninterpretable Gij vector {exchange_g[i]}')
                this = to_cartesian(exchange_g[i])
                j, k = self.operators[self.jindices[i]]
                gval = np.array([j[1] @ k[2] - j[2] @ k[1], j[2] @ k[0] - j[0] @ k[2], j[0] @ k[1] - j[1] @ k[0]])
                ham0 += np.einsum('i,ijk', this, gval)
        # tensor (anisotropic) exchange (field independent)
        if len(exchange_d) > 0:
            for i in range(len(self.jindices)):
                j, k = self.operators[self.jindices[i]]
                this = exchange_d[i]
                if len(this) == 0:  # no zero field splitting (isotropic); i.e. [] -> [0,0,0]
                    continue
                elif len(this) == 1:  # axial D without rotation; e.g. [3.0] = [-1.0,-1.0,2.0]
                    this = this[0] * np.array([-1 / 3, -1 / 3, 2 / 3])
                    contrib = np.einsum('uaw,uwb->uab', j, k)
                    contrib = np.einsum('u,uvw', this, contrib)
                elif len(this) == 2:  # rhombic D w/o rotation [D,E/D]; e.g. [3.0,0.333333] = [-2.0,0.0,2.0]
                    this = this[0] * np.array([-1 / 3 + this[1], -1 / 3 - this[1], 2 / 3])
                    contrib = np.einsum('uaw,uwb->uab', j, k)
                    contrib = np.einsum('u,uvw', this, contrib)
                elif len(this) == 3:  # rhombic D, full diagonal specified; e.g. [1,0,-1] = [1,0,-1]
                    # We are not forcing tracelessness so that isotropic J can be rolled into this
                    # this = np.array(this) - sum(this)/3
                    contrib = np.einsum('uaw,uwb->uab', j, k)
                    contrib = np.einsum('u,uvw', this, contrib)
                elif len(this) == 5 or len(this) == 6:  # rhombic + Euler angles [D,E/D,alpha,beta,gamma]
                    this = get_d_mat(this)
                    contrib = np.einsum('uaw,uv,vwb', j, this, k)
                else:  # improperly formatted
                    raise ValueError(f'Uninterpretable Dij tensor: {this}')
                ham0 += contrib
        return Hamiltonian(ham0, ham1)

    def expectation(self, state) -> np.ndarray:
        """Return spin expectation value at each center for a given state"""
        conj = np.conjugate(state)
        return np.real([conj @ ops @ state for ops in self.operators])


class IntegrationGrid:
    """Contains a grid of points used for integration"""

    def __init__(self, gridtype, deg=0, domain='sphere'):
        """
        Constructs the integration grid
        'gaussian': accepts precisions >= 3, defaults to 5, and can cover domains
                    of a full sphere, a hemisphere, and an octant
        'lebedev':  accepts any valid lebedev precision, defaults to 19, and can
                    adapt the grid to a sphere, hemisphere, or octant
        'custom':   uses a user-defined grid; ignores the 'deg'/precision; takes
                    the 'domain' value as a filename from which to read the grid;
                    file should be a CSV file with rows [theta, phi, weight] using
                    units of degrees and not radians
        """
        self.points = []
        self.source = None
        if gridtype not in ['gaussian', 'lebedev', 'custom']:
            raise ValueError(f'Unrecognized/unsupported integration method: {gridtype!r}')
        if gridtype == 'gaussian' or gridtype == 'lebedev':
            if domain not in ['octant', 'hemisphere', 'sphere']:
                raise ValueError(f'Unrecognized integration domain {domain!r}')
        if gridtype == 'gaussian':
            if isinstance(deg, int):
                deg = [deg, deg] if deg > 0 else [5, 5]
            if not isinstance(deg, list):
                raise ValueError(f'Uninterpretable grid precision of {deg}')
            if len(deg) == 1:
                deg = [deg[0], deg[0]]
            if len(deg) != 2:
                raise ValueError(f'Unusual list length in Gaussian grid {deg}')
            if any([not isinstance(x, int) for x in deg]):
                raise ValueError(f'Grid precision must be an integer but {deg} contains non-integers')
            if any([x < 3 for x in deg]):
                raise ValueError(f'Minimum Gaussian precision is 3 but {deg} has been supplied')
            for rx, wx in np.transpose(np.polynomial.legendre.leggauss(deg[0])):  # `x` corresponds to azimuthal `theta`
                theta = (np.pi / 4) * (rx + 1)  # adjust from unit sphere to octant 0..pi/2
                ct, st = np.cos(theta), np.sin(theta)
                for ry, wy in np.transpose(np.polynomial.legendre.leggauss(deg[1])):  # `y` corresponds to polar `phi`
                    phi = (np.pi / 4) * (ry + 1)  # adjust from unit sphere to octant 0..pi/2
                    cf, sf = np.cos(phi), np.sin(phi)
                    if domain == 'octant':  # if octant requested, write current grid
                        self.points.append([ st * cf, st * sf, ct, st * wx * wy * np.pi / 8])
                    elif domain == 'hemisphere':  # if hemisphere, copy current grid onto four octants
                        self.points.append([ st * cf, st * sf, ct, st * wx * wy * np.pi / 32])
                        self.points.append([ st * cf,-st * sf, ct, st * wx * wy * np.pi / 32])
                        self.points.append([-st * cf, st * sf, ct, st * wx * wy * np.pi / 32])
                        self.points.append([-st * cf,-st * sf, ct, st * wx * wy * np.pi / 32])
                    elif domain == 'sphere':  # if sphere, copy current grid onto every octant
                        self.points.append([ st * cf, st * sf, ct, st * wx * wy * np.pi / 64])
                        self.points.append([ st * cf, st * sf,-ct, st * wx * wy * np.pi / 64])
                        self.points.append([ st * cf,-st * sf, ct, st * wx * wy * np.pi / 64])
                        self.points.append([ st * cf,-st * sf,-ct, st * wx * wy * np.pi / 64])
                        self.points.append([-st * cf, st * sf, ct, st * wx * wy * np.pi / 64])
                        self.points.append([-st * cf, st * sf,-ct, st * wx * wy * np.pi / 64])
                        self.points.append([-st * cf,-st * sf, ct, st * wx * wy * np.pi / 64])
                        self.points.append([-st * cf,-st * sf,-ct, st * wx * wy * np.pi / 64])
        elif gridtype == 'lebedev':
            leb_grids = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83,
                         89, 95, 101, 107, 113, 119, 125, 131]
            if isinstance(deg, list) and len(deg) == 1:  # switch [n] -> n
                deg = deg[0]
            if not isinstance(deg, int):
                raise ValueError(f'Unusual Lebedev grid specification of {deg}; expected integer not {type(deg)}')
            if deg == 0:
                deg = 19
            if deg not in leb_grids:
                raise ValueError(f'Unrecognized Lebedev precision of {deg}; we only support {leb_grids}')
            # Now we need to locate the grid files...
            try:  # First, check for an environment variable LEBEDEV
                leb_folder = environ['LEBEDEV']
            except KeyError:  # Otherwise look for 'lebedev' subdirectory of curr dir
                leb_folder = 'lebedev'
            leb = np.loadtxt(leb_folder + f'/grid_{deg:03d}.csv', delimiter=',')
            if domain == 'sphere':
                for i in leb:
                    self.points.append(i[[0, 1, 2, 3]])
            elif domain == 'hemisphere':
                for i in leb:
                    if i[4]:  # if i[4] is zero then skip
                        self.points.append(i[[0, 1, 2, 4]])
            elif domain == 'octant':
                for i in leb:
                    if i[5]:  # if i[5] is zero then skip
                        self.points.append(i[[0, 1, 2, 5]])
        else:  # 'custom'
            if not isinstance(domain, str):
                raise ValueError(f'Custom grid expects a filename in \'domain\' but user supplied {domain} of '
                                 f'type {type(domain)}')
            deg = 0
            pts = np.loadtxt(domain, delimiter=',')
            for i in pts:
                theta, phi, w = i[0] * np.pi / 180, i[1] * np.pi / 180, i[2]
                ct, st = np.cos(theta), np.sin(theta)
                cf, sf = np.cos(phi), np.sin(phi)
                self.points.append([st * cf, st * sf, ct, w])
            self.source = domain
            domain = 'sphere'
        self.size = len(self.points)
        self.points = np.array(self.points)
        self.method = gridtype
        self.domain = domain
        self.precision = deg


def integrate(temperature, field, spinsystem, variable_values, grid, cutoff=0.001) -> np.ndarray:
    """Integrates -<S>*Z^T over a grid, either the full 3x3 matrix or just its diagonal, depending on the symmetry of
    the grid supplied. If abs(field) is too small (less than `cutoff`), an intensity of zero will be automatically
    returned because as field approaches/equals 0, numerical instabilities result in spurious predictions of intensity.

    :param temperature: Temperature (Kelvin) as a real positive float
    :param field: Field (Tesla) as a real float
    :param spinsystem: `SpinSystem` variable
    :param variable_values: List of values for resolving variables
    :param grid: `IntegrationGrid` variable
    :param cutoff: Real positive float describing smallest field used in simulation to avoid numerical instabilities.
    :return: list of integrated matrices (or matrix diagonals), one for each spin center
    """
    dim = 3 if grid.domain == 'octant' else 9  # polarization dimension
    integration = np.zeros((spinsystem.nspins, dim))
    if abs(field) < cutoff:
        return integration  # just return `sum` (zero currently) intensity
    ham = spinsystem.ham(*variable_values)
    for pt in grid.points:
        contrib = np.zeros((spinsystem.nspins, 3))
        eigenvalues, eigenvectors = np.linalg.eigh(ham.at_field(field * np.array(pt[:3])))
        eigenvalues = eigenvalues - eigenvalues.min()  # shift lowest energy to 0 to prevent overflows
        # now we do exp(-E/(kB*T)), where -1.4387768775039338 is -1/kB in K/cm-1
        eigenvalues = np.exp((-1.4387768775039338 / temperature) * eigenvalues)
        eigenvalues = eigenvalues / eigenvalues.sum()  # exp(-E/(kB*T)) / sum_k[exp(-E_k/(kB*T))]
        # the variable `eigenvalues' now contains fractional Boltzmann populations instead of energies
        for i in range(len(eigenvalues)):
            if eigenvalues[i] == 0.0:  # if population underflows, don't bother
                continue
            contrib += eigenvalues[i] * spinsystem.expectation(eigenvectors[:, i])
        if dim == 3:
            # Mix contrib ([Sx, Sy, Sz]) & pt[:3] ([X, Y, Z]) -> [SxX, SyY, SzZ] for each spin center
            contrib = contrib @ np.diag(pt[:3])
        else:
            # Mix contrib ([Sx, Sy, Sz]) & pt[:3] ([X, Y, Z]) -> [SxZx, SxZy, SxZz, SyZx, SyZy, SyZz, SzZx, SzZy, SzZz]
            # for each spin center. This is equivalent to np.kron(contrib, pt[:3]) but np.einsum+reshape is faster!
            contrib = np.einsum('ij,k', contrib, pt[:3]).reshape(contrib.shape[0], 9)
        integration += pt[3] * contrib
    return -integration  # the negative is because this is -<S>*Z^T
