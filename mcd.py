# Be sure to use Python 3.5-3.7 after installing pyswarms and pyyaml 
# e.g. install with: python3.6 -m pip install pyswarms, pyyaml 
import numpy as np 
from scipy.linalg import eig, svd, inv, pinv 
from scipy.special.orthogonal import p_roots 
from scipy.optimize import curve_fit 
from yaml import safe_load, YAMLError 
import sys, os, csv, time 
import pyswarms as ps 

# some initial useful constants, etc. 
zeemanFactor = 0.4668644735835207  # Bohr magneton in units of cm-1/Tesla 
boltzFactor = -1.4387768775039338  # -1/kB in units of Kelvin/cm-1
htFactor = 0.33585690476000785     # (Bohr mag)/(2kB) in units of Kelvin/Tesla 
gen_pol_mat = [['M'+str(i+1)] for i in range(3)] 
three_pol_mat = [['Myz'], ['Mzx'], ['Mxy']] 
numeric = (int, float, np.int16, np.int32, np.int64, np.uint16, np.uint32, 
           np.uint64, np.float16, np.float32, np.float64) 

""" express_duration """ 
def express_duration(start_time, separator='\n'): 
  now = time.time() 
  diff = now-start_time 
  out = "" 
  if diff >= 86400: 
    out += str(int(diff//86400))+" days " 
    diff = diff % 86400 
  if diff >= 3600: 
    out += str(int(diff//3600))+" h " 
    diff = diff % 3600 
  if diff >= 60: 
    out += str(int(diff//60))+" min " 
    diff = diff % 60 
  out += ('%.3f'%diff)+" sec" 
  print(separator + 'Job successfully completed in', out, flush=True) 

""" raise_error: Prints the requested string to the user as an error then quits 
    by returning 1 (generic error code). """
def raise_error(*args): 
  print('ERROR:',*args) 
  sys.exit(1) 

""" sf: formats numbers using significant figures 
    converts a number 'x' into a string with 'sig' significant figures and 
    rounds anything less than 1e-10 in magnitude to zero """ 
def sf(x, sig=6): 
  if isinstance(x, (list, np.ndarray)): # if list 
    return '[' + ', '.join([sf(n, sig) for n in x]) + ']' 
  elif isinstance(x, numeric): # if valid number 
    if np.isnan(x): # if NaN, return 'NaN' string 
      return 'NaN' 
    x = float(x) 
  else: # if invalid 'x', use Python's default str() 
    return str(x) 
  if not isinstance(sig, int): # if invalid 'sig', use Python's default str()
    return str(x) 
  elif sig <= 0: 
    return str(x) 
  if abs(x) < 1e-10: # if -1e-10 < x < 1e-10, report as 0 
    n = ''.zfill(sig) 
    return n[:1]+'.'+n[1:] 
  elif np.sign(x) < 0: # if 'x' is negative 
    return '-'+sf(abs(x), sig) 
  else: # x is positive and non-negligible, so we will format... 
    q=int(np.floor(np.log10(x))) 
    n = str(round(x*10**(sig-1-q))) # only keeping 'sig' sig figs 
    if len(n) != sig: # if rounding changes # digits
      x = round(x*10**(sig-1-q))
      x = x*10**(q-sig+1) 
      return sf(x, sig) 
    # when to use engineering notation 
    if q > 5 or q < -5: # <1e-5 or >=1e6 
      n = sf(x*10**(-q), sig) 
      if q > 0: 
        n += 'e+'+str(q) 
      else: 
        n += 'e'+str(q) 
      return n 
    # otherwise 
    else: 
      if q+1 >= sig: 
        n = n.ljust(q+1,'0') # add appropriate number of 0s if missing 
      elif q >= 0 and q+1 < sig: 
        n = n[:q+1]+'.'+n[q+1:] 
      else: 
        n = '0.'.ljust(1-q,'0') + n 
      return n 

""" number_error: formats numbers with error 
    Formats "3.04 +/- 0.02" as "3.04(2)" and handles a few exceptions """ 
def number_error(x, s): 
  if not isinstance(x, numeric) or not isinstance(s, numeric): 
    return str(x) + ' +/- ' + str(s) 
  if np.isnan(x): 
    return 'NaN' 
  elif np.isnan(s): 
    return sf(x)+'(NaN)' 
  if abs(x) < 1e-10: # round small x to zero 
    x = 0 
  if s < 1e-10: # small or negative error -> no error 
    s = 0 
  if s == 0: # without error, just show with default number of sig figs 
    return sf(x) 
  if x < 0: # if negative, add minus sign 
    return '-'+number_error(abs(x), s) 
  r = int(np.floor(np.log10(s))) # magnitude of s 
  x = round(x*10**(-r))*10**(r) # round x as appropriate 
  if abs(x) > 1e-10: # if still nonzero 
    q = int(np.floor(np.log10(x))) # magnitude of x 
  else: 
    if r <= 5 and r >= 0: 
      n = str(int(s*10**(-r))*10**(r)) # format error with zeros 
      return '0('+n+')' 
    elif r < 0 and r >= -5: 
      m = ''.zfill(1-r) # show appropriate number of zeros 
      m = m[:1]+'.'+m[1:] # add decimal point 
      n = str(int(s*10**(-r))) # keep only first digit of 's' 
      return m+'('+n+')' 
    q = -12 # say x is very small 
  # when to use engineering notation 
  if max(q,r) > 5 or max(q,r) < -5: # <1e-5 or >=1e6 
    m = str(round(x*10**(-r))) # keep sig figs of x up to 'r' 
    n = str(int(s*10**(-r))) # keep only first digit of 's' 
    n = '('+n+')' # add parentheses 
    if q-r > 1: # if there's at least one more x sig fig than s 
      m = m[:1]+'.'+m[1:] # add a decimal point 
    m += n 
    if max(q,r) > 0: 
      m += 'e+'+str(max(q,r)) 
    else: 
      m += 'e'+str(max(q,r)) 
    return m 
  # if no engineering notation 
  if r > 0: 
    n = str(int(s*10**(-r))*10**(r)) # reformat with zeros 
  else: 
    n = str(int(s*10**(-r))) # keep only first digit of 's' 
  return sf(x, q-r+1)+'('+n+')' 

""" print_matrix_eq: print a matrix equation 
    - accepts a list of matrices and non-matrix strings (like '=') """ 
def print_matrix_eq(*list_of_matrices, indent=6): 
  formatted = []
  for mat in list_of_matrices: 
    if isinstance(mat, (list, np.ndarray)): 
      mat = np.transpose(mat) 
      formatted.append(['[']*len(mat[0])) 
      for col in mat: 
        formatted.append([sf(x) for x in col]) 
        formatted.append([' ']*len(col)) 
      formatted.pop() 
      formatted.append([']']*len(mat[0])) 
    elif isinstance(mat, str): 
      formatted.append([mat]) 
  nrows = max([len(x) for x in formatted]) 
  list_of_matrices = list() 
  for mat in formatted: 
    x = [sf(y) for y in mat] 
    x = x+['']*(nrows-len(x)) 
    maxlen = max([len(y) for y in x]) 
    x = [y+' '*(maxlen-len(y)) for y in x] 
    list_of_matrices.append(x) 
  list_of_matrices = list(map(list, zip(*list_of_matrices))) # transpose 
  for row in list_of_matrices: 
    print(' '*indent+''.join(row))

""" IntegrationGrid class: contains a grid of points used for integration """ 
class IntegrationGrid: 
  def __init__(self, gridtype, deg, domain='auto'): 
    self.grid = [] 
    if gridtype is None: 
      print('- Unspecified method; defaulting to Gaussian quadrature') 
      gridtype = 'gaussian' 
    if gridtype not in ['gaussian', 'lebedev', 'discrete']: 
      raise_error('Unrecognized/unsupported integration method:',str(gridtype)) 
    if domain not in ['auto', 'octant', 'hemisphere', 'sphere']: 
      raise_error('Unrecognized integration domain:', str(domain)) 
    if domain == 'auto': domain = 'octant' 
    if gridtype == "gaussian": 
      if not isinstance(deg, (list, int)): 
        raise_error('Uninterpretable grid precision') 
      if isinstance(deg, int): 
        if deg == 0: deg = 5 
        if deg < 3: raise_error('Minimum Gaussian precision is 3') 
        deg = [deg] 
      if len(deg) == 1: 
        deg = [deg[0], deg[0]] 
      elif len(deg) != 2: 
        raise_error('Unusual list length in Gaussian grid:', str(deg)) 
      [rx,wx] = p_roots(deg[0]) 
      [ry,wy] = p_roots(deg[1]) 
      for i in range(len(rx)): # i corresponds to theta (0..pi/2) 
        theta = (np.pi/4.0)*(rx[i]+1.0) # adjust region from unit square 
        ct = np.cos(theta) 
        st = np.sin(theta) 
        for j in range(len(ry)): # j correspond to phi (0..pi/2) 
          phi = (np.pi/4.0)*(ry[j]+1.0) # adjust region from unit square 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          if domain == 'octant': 
            self.grid.append([st*cf, st*sf, ct, st*wx[i]*wy[j]*(np.pi/8)]) 
          elif domain == 'hemisphere': 
            self.grid.append([ st*cf,  st*sf, ct, st*wx[i]*wy[j]*(np.pi/32)]) 
            self.grid.append([-st*cf,  st*sf, ct, st*wx[i]*wy[j]*(np.pi/32)]) 
            self.grid.append([ st*cf, -st*sf, ct, st*wx[i]*wy[j]*(np.pi/32)]) 
            self.grid.append([-st*cf, -st*sf, ct, st*wx[i]*wy[j]*(np.pi/32)]) 
          elif domain == 'sphere': 
            self.grid.append([ st*cf,  st*sf,  ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([ st*cf,  st*sf, -ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([ st*cf, -st*sf,  ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([-st*cf,  st*sf,  ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([-st*cf, -st*sf,  ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([-st*cf,  st*sf, -ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([ st*cf, -st*sf, -ct, st*wx[i]*wy[j]*(np.pi/64)]) 
            self.grid.append([-st*cf, -st*sf, -ct, st*wx[i]*wy[j]*(np.pi/64)]) 
    elif gridtype == "lebedev": 
      leb_grids = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 
                   41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 
                   125, 131] 
      if not isinstance(deg, int): 
        raise_error('Lebedev: Unusual Lebedev grid specification of '+str(deg)) 
      elif deg not in leb_grids: 
        raise_error('Lebedev: Unrecognized Lebedev precision of '+str(deg)) 
      if deg == 0: 
        print('- Using minimum Lebedev precision of 3')
        deg = 3 
      # Now we need to locate the grid files... 
      try: # First, check for an environment variable LEBEDEV 
        leb_folder = os.environ['LEBEDEV'] 
      except KeyError: # Otherwise look for 'lebedev' subdirectory of curr dir 
        leb_folder = 'lebedev' 
      try: # Now we try to import the file 
        leb = np.loadtxt(leb_folder+'/leb_%03d.csv'% deg, delimiter=',') 
      except OSError: # If file not found 
        raise_error('Lebedev grid file leb_%03d.csv not found\n'% deg, 
                    '       check LEBEDEV environment variable') 
      if domain == 'sphere': 
        for i in leb: 
          theta, phi, w = i[0]*np.pi/180.0, i[1]*np.pi/180.0, i[2] 
          ct = np.cos(theta) 
          st = np.sin(theta) 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          self.grid.append([st*cf, st*sf, ct, w]) 
      elif domain == 'hemisphere': 
        for i in leb: 
          theta, phi, w = i[0]*np.pi/180.0, i[1]*np.pi/180.0, i[2] 
          ct = np.cos(theta) 
          st = np.sin(theta) 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          if i[0] == 90: # if on equator, add point 
            self.grid.append([st*cf, st*sf, ct, w]) 
          elif i[0] < 90: # if in upper hemisphere double weight 
            self.grid.append([st*cf, st*sf, ct, 2*w]) 
      elif domain == 'octant': 
        for i in leb: 
          theta, phi, w = i[0]*np.pi/180.0, i[1]*np.pi/180.0, i[2] 
          ct = np.cos(theta) 
          st = np.sin(theta) 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          if 0 <= i[0] <= 90 and 0 <= i[1] <= 90: # if in first octant 
            edge_azimuthal = (i[0] == 0 or i[0] == 90) 
            edge_polar =     (i[1] == 0 or i[1] == 90) 
            if edge_azimuthal and edge_polar: # and (on two edges ie vertex) 
              self.grid.append([st*cf, st*sf, ct, 2*w]) 
            elif edge_azimuthal or edge_polar: # xor (on just one edge) 
              self.grid.append([st*cf, st*sf, ct, 4*w]) 
            else: # not on an edge 
              self.grid.append([st*cf, st*sf, ct, 8*w]) 
    elif gridtype == "discrete": 
      domain = 'auto' 
      if not isinstance(deg, (list, str)): 
        raise_error('Specify file or list discrete [theta,phi] in radians\n', 
                    '       Error interpreting', str(deg)) 
      if isinstance(deg, str): # if we're supplied with a file name 
        pts = np.loadtxt(deg) # read grid from file; format: [phi,theta,w] 
        rx = pts[:,1]*np.pi/180.0; ry = pts[:,0]*np.pi/180.0; w = pts[:,2] 
        for i in range(len(w)): 
          theta = rx[i]; phi = ry[i] 
          ct = np.cos(theta) 
          st = np.sin(theta) 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          self.grid.append([st*cf, st*sf, ct, w[i]]) 
      else: 
        if len(deg) == 0: 
          raise_error('Empty list of discrete orientations provided')
        if len(deg) == 2: 
          if all(isinstance(x, numeric) for x in deg): 
            deg = [deg] 
        num = len(deg) 
        for i in deg: 
          if not isinstance(x, list): 
            raise_error('Discrete grid non-list error for', str(i), 
                        'in', str(deg)) 
          if len(x) != 2: 
            raise_error('Discrete grid accepts two element lists, not', str(i), 
                        'in', str(deg)) 
          if not all(isinstance(j, numeric) for j in i): 
            raise_error('Discrete grid type error for', str(i), 'in', str(deg)) 
          [phi, theta] = np.array(i)*np.pi/180.0 
          ct = np.cos(theta) 
          st = np.sin(theta) 
          cf = np.cos(phi) 
          sf = np.sin(phi) 
          self.grid.append([st*cf, st*sf, ct, 1.0/num]) 
    self.size = len(self.grid) 
    self.method = gridtype 
    self.domain = domain 
    self.deg = deg 

""" euler_rotate: convert zyz Euler angles into a rotation matrix """ 
def euler_rotate(alpha, beta, gamma): 
  c1 = np.cos(alpha) 
  s1 = np.sin(alpha) 
  c2 = np.cos(beta) 
  s2 = np.sin(beta) 
  c3 = np.cos(gamma) 
  s3 = np.sin(gamma) 
  a11 =  c1*c2*c3-s1*s3 
  a12 =  c1*s3+c2*c3*s1 
  a13 = -c3*s2 
  a21 = -c3*s1-c1*c2*s3 
  a22 =  c1*c3-c2*s1*s3 
  a23 =  s2*s3 
  a31 =  c1*s2 
  a32 =  s1*s2 
  a33 =  c2 
  return np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]) 

""" get_g_mat: construct 3x3 g value matrix from a 6-membered list 
    [gx, gy, gz, alpha, beta, gamma] using zyz Euler angles """
def get_g_mat(gvalueinfo): 
  gx = gvalueinfo[0] 
  gy = gvalueinfo[1] 
  gz = gvalueinfo[2] 
  alpha = gvalueinfo[3] 
  beta  = gvalueinfo[4] 
  gamma = gvalueinfo[5] 
  rot = euler_rotate(alpha,beta,gamma) 
  rotT = np.transpose(rot) 
  g = [[gx,0,0],[0,gy,0],[0,0,gz]] 
  g = np.dot(g,rotT) 
  g = np.dot(rot,g) 
  return np.array(g) 

""" get_d_mat: construct 3x3 ZFS matrix from a 5-membered list 
    [D, E/D, alpha, beta, gamma] using zyz Euler angles """
def get_d_mat(dvalueinfo): 
  d   = dvalueinfo[0] 
  eod = dvalueinfo[1] 
  alpha = dvalueinfo[2] 
  beta  = dvalueinfo[3] 
  gamma = dvalueinfo[4] 
  rot = euler_rotate(alpha,beta,gamma) 
  rotT = np.transpose(rot) 
  dmat = [[(-1/3.0)*d+eod*d,0,0],[0,(-1/3.0)*d-eod*d,0],[0,0,(2/3.0)*d]] 
  dmat = np.dot(dmat,rotT) 
  dmat = np.dot(rot,dmat) 
  return np.array(dmat) 

""" resolve_variables: converts from supplied variable values to usable 
    magnetic parameters """ 
def resolve_variables(x_vals, x_vars, vars): 
  x_out = [] 
  for i in range(len(x_vals)): 
    if isinstance(x_vals[i], list): # if it's a list 
      x_out.append(resolve_variables(x_vals[i], x_vars[i], vars)) 
    elif isinstance(x_vals[i], str): # if it's a variable name  
      x_out.append(vars[x_vars[i]]) 
    else: # if it's a number 
      x_out.append(x_vals[i]) 
  return x_out 

""" Hamiltonian class: separately stores field-independent and field-dependent 
    pieces of Ham matrix for easy recalculation at each point in the grid """
class Hamiltonian: 
  def __init__(self, ham0, hamx, hamy, hamz): 
    self.indep = np.array(ham0) 
    self.dep = np.array([hamx, hamy, hamz]) 
  def at_field(self, field): 
    # supply field as a three-membered list [Bx, By, Bz] 
    return self.indep + np.tensordot(field, self.dep, (0,0)) 

""" SpinSystem class: contains info on magnetic parameters and variables 
    - Constructs 'Hamiltonian' objects when supplied with variable values 
    - Reports spin expectation values when supplied with a mixed state """
class SpinSystem: 
  def __init__(self, spinlist): 
    def mult(spin): # calculates multiplicity of a spin, returning an int 
      return int(round(2*spin+1)) 
    def spinX(spin): # construct a spin X matrix 
      mat = [] 
      m = float(spin) 
      while m >= -spin: 
        matrow = [] 
        n = float(spin) 
        while n >= -spin: 
          if m==n+1 or m+1==n: 
            matrow.append(np.sqrt(spin*(spin+1)-m*n)/(2.0)) 
          else: 
            matrow.append(0.0) 
          n = n-1 
        mat.append(matrow) 
        m = m-1 
      return mat 
    def spinY(spin): # construct a spin Y matrix 
      mat = [] 
      m = float(spin) 
      while m >= -spin: 
        matrow = [] 
        n = float(spin) 
        while n >= -spin: 
          if m==n+1 or m+1==n: 
            matrow.append((m-n)*np.sqrt(spin*(spin+1)-m*n)/(2j)) 
          else: 
            matrow.append(0.0) 
          n = n-1 
        mat.append(matrow) 
        m = m-1 
      return mat 
    def spinZ(spin): # construct a spin Z matrix 
      mat = [] 
      m = float(spin) 
      while m >= -spin: 
        matrow = [] 
        n = float(spin) 
        while n >= -spin: 
          if m==n: 
            matrow.append(float(n)) 
          else: 
            matrow.append(0.0) 
          n = n-1 
        mat.append(matrow) 
        m = m-1 
      return mat 
    def kron(mats): # performs Kronecker product 
      if len(mats)==0: 
        return mats 
      elif len(mats)==1: 
        return mats[0] 
      elif len(mats)==2: 
        return np.kron(mats[0],mats[1]) 
      else: 
        return kron([np.kron(mats[0],mats[1])]+mats[2:]) 
    
    # make composite Cartesian spin operator matrices with Kronecker products 
    # first, SX matrices 
    mat = [] 
    for m in range(len(spinlist)): 
      matrow = [] 
      for n in range(len(spinlist)): 
        if m==n: 
          matrow.append(spinX(spinlist[n])) 
        else: 
          matrow.append(np.identity(mult(spinlist[n]))) 
      mat.append(matrow) 
    sx = np.array(list(map(kron,mat))) 
    # second, SY matrices 
    mat = [] 
    for m in range(len(spinlist)): 
      matrow = [] 
      for n in range(len(spinlist)): 
        if m==n: 
          matrow.append(spinY(spinlist[n])) 
        else: 
          matrow.append(np.identity(mult(spinlist[n]))) 
      mat.append(matrow) 
    sy = np.array(list(map(kron,mat))) 
    # third, SZ matrices 
    mat = [] 
    for m in range(len(spinlist)): 
      matrow = [] 
      for n in range(len(spinlist)): 
        if m==n: 
          matrow.append(spinZ(spinlist[n])) 
        else: 
          matrow.append(np.identity(mult(spinlist[n]))) 
      mat.append(matrow) 
    sz = np.array(list(map(kron,mat))) 
    # make a list of indices to keep exchange coupling in order 
    mat = [] 
    for i in range(len(spinlist)): 
      for j in range(i+1,len(spinlist)): 
        mat.append([i,j]) # ordered combinations of nonidentical spins 
    # set variables 
    self.jindex = np.array(mat) 
    self.spins = list(spinlist) 
    self.num = len(spinlist) 
    self.SX = sx 
    self.SY = sy 
    self.SZ = sz 
    self.S = np.array([sx, sy, sz]) 
    self.zero = 0*sz[0] 
    self.vars = None 
  
  def set_grid(self, intgrid): 
    self.grid = intgrid 
  
  def set_mag_values(self, g_vals, d_vals, j_vals, eeG_vals, eeD_vals, jfact): 
    self.g_vals = g_vals 
    self.d_vals = d_vals 
    self.j_vals = j_vals 
    self.eeG_vals = eeG_vals 
    self.eeD_vals = eeD_vals 
    self.jfactor = jfact 
  
  def set_mag_variables(self, g_vars, d_vars, j_vars, eeG_vars, eeD_vars): 
    self.g_vars = g_vars 
    self.d_vars = d_vars 
    self.j_vars = j_vars 
    self.eeG_vars = eeG_vars 
    self.eeD_vars = eeD_vars 
  
  def initiate_variables(self, var_names, var_bounds): 
    self.var_names = var_names 
    self.var_bounds = var_bounds 
    self.vars = [bound.mean() for bound in var_bounds] 
  
  def set_vars(self, vars): 
    self.vars = vars 
  
  def ham(self, vars=None): # makes Hamiltonian if no unresolved variables 
    if vars is None: # if not provided with a new list of variables, use old 
      vars = self.vars 
    glist = resolve_variables(self.g_vals, self.g_vars, vars) 
    dlist = resolve_variables(self.d_vals, self.d_vars, vars) 
    jlist = resolve_variables(self.j_vals, self.j_vars, vars) 
    eeGlist = resolve_variables(self.eeG_vals, self.eeG_vars, vars) 
    eeDlist = resolve_variables(self.eeD_vals, self.eeD_vars, vars) 
    
    ham0 = np.array(self.zero) 
    hamx = np.array(self.zero) 
    hamy = np.array(self.zero) 
    hamz = np.array(self.zero) 
    
    # construct Zeeman perturbation (field-dependent) 
    for i in range(self.num): 
      # isotropic g; e.g. [2.0] = [2.0,2.0,2.0] 
      if len(glist[i]) == 1: 
        hamx = hamx + zeemanFactor*glist[i][0]*self.SX[i] 
        hamy = hamy + zeemanFactor*glist[i][0]*self.SY[i] 
        hamz = hamz + zeemanFactor*glist[i][0]*self.SZ[i] 
      # axial g without rotation; e.g. [2.0,2.1] = [2.0,2.0,2.1] 
      elif len(glist[i])==2: 
        hamx = hamx + zeemanFactor*glist[i][0]*self.SX[i] 
        hamy = hamy + zeemanFactor*glist[i][0]*self.SY[i] 
        hamz = hamz + zeemanFactor*glist[i][1]*self.SZ[i] 
      # rhombic g without rotation; e.g. [2.0,2.1,2.2] 
      elif len(glist[i])==3: 
        hamx = hamx + zeemanFactor*glist[i][0]*self.SX[i] 
        hamy = hamy + zeemanFactor*glist[i][1]*self.SY[i] 
        hamz = hamz + zeemanFactor*glist[i][2]*self.SZ[i] 
      # rhombic with rotation by Euler angles as [gx,gy,gz,alpha,beta,gamma] 
      # or full 3x3 matrix [gxx,gxy,gxz,gyx,gyy,gyz,gzx,gzy,gzz] 
      elif len(glist[i])==6 or len(glist[i])==9: 
        if len(glist[i])==6: 
          gmat = get_g_mat(glist[i]) 
        else: 
          gmat = np.array(glist[i]).reshape((3,3)) 
        for j in range(3): 
          hamx = hamx + zeemanFactor*gmat[0][j]*self.S[j][i] 
          hamy = hamy + zeemanFactor*gmat[1][j]*self.S[j][i] 
          hamz = hamz + zeemanFactor*gmat[2][j]*self.S[j][i] 
      # unrecognized format 
      else: 
        raise_error('Uninterpretable g tensor:', str(glist[i])) 
    # construct zero-field splitting perturbation (field-independent) 
    for i in range(self.num): 
      # no zero field splitting; i.e. []
      if len(dlist[i])==0: 
        continue 
      # axial D without rotation; e.g. [3.0] = [-1.0,-1.0,2.0] 
      elif len(dlist[i])==1: 
        dval = float(dlist[i][0]) 
        dmat = (-np.dot(self.SX[i],self.SX[i]) 
                -np.dot(self.SY[i],self.SY[i]) 
                +2.0*np.dot(self.SZ[i],self.SZ[i]))
        ham0 = ham0 + (dval/3.0)*dmat 
      # rhombic D w/o rotation [D,E/D]; e.g. [3.0,0.333333] = [-2.0,0.0,2.0] 
      elif len(dlist[i])==2: 
        dval = float(dlist[i][0]) 
        eod = float(dlist[i][1]) 
        dmat = dval*np.array([-1/3+eod,-1/3-eod,2/3]) 
        ham0 = ham0 + (dmat[0]*np.dot(self.SX[i],self.SX[i]) 
                      +dmat[1]*np.dot(self.SY[i],self.SY[i]) 
                      +dmat[2]*np.dot(self.SZ[i],self.SZ[i])) 
      # rhombic + rotate by Euler angles [D,E/D,alpha,beta,gamma] 
      # or full 3x3 matrix: [Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz] 
      elif len(dlist[i])==5 or len(dlist[i])==9: 
        if len(dlist[i])==5: 
          dmat = get_d_mat(dlist[i]) 
        else: 
          dmat = np.array(dlist[i]).reshape((3,3)) 
        for j in range(3): 
          for k in range(3): 
            ham0 = ham0+dmat[j][k]*np.dot(self.S[j][i],self.S[k][i]) 
      # unrecognized format 
      else: 
        raise_error('Uninterpretable D tensor:', str(dlist[i])) 
    # construct isotropic J coupling perturbation (field-independent) 
    if len(jlist)>0: 
      for i in range(len(self.jindex)): 
        j,k = self.jindex[i] 
        jmat = (np.dot(self.SX[j],self.SX[k]) 
               +np.dot(self.SY[j],self.SY[k]) 
               +np.dot(self.SZ[j],self.SZ[k])) 
        ham0 = ham0 + self.jfactor*jlist[i]*jmat 
    # construct antisymmetric vector G perturbation (field-independent) 
    if len(eeGlist)>0: 
      for i in range(len(self.jindex)): 
        j,k = self.jindex[i] 
        ct,st = [np.cos(eeGlist[i][1]), np.sin(eeGlist[i][1])] 
        cf,sf = [np.cos(eeGlist[i][2]), np.sin(eeGlist[i][2])] 
        eeGx,eeGy,eeGz = eeGlist[i][0]*np.array([cf*st,sf*st,ct]) 
        eeG = np.array([np.dot(self.SY[j],self.SZ[k])
                        -np.dot(self.SZ[j],self.SY[k]), 
                        np.dot(self.SZ[j],self.SX[k])
                        -np.dot(self.SX[j],self.SZ[k]), 
                        np.dot(self.SX[j],self.SY[k])
                        -np.dot(self.SY[j],self.SX[k])])  
        ham0 = ham0 + eeGx*eeG[0]+eeGy*eeG[1]+eeGz*eeG[2] 
    # construct anisotropic tensor D perturbation (field-independent) 
    if len(eeDlist)>0: 
      for i in range(len(self.jindex)): 
        j,k = self.jindex[i] 
        # no dipole-dipole coupling; i.e. []
        if len(eeDlist[i])==0: 
          continue 
        # axial eeD w/o rotation; e.g. [3.0] = [-1.0,-1.0,2.0] 
        elif len(eeDlist[i])==1: 
          dval = float(eeDlist[i][0]) 
          dmat = (-np.dot(self.SX[j],self.SX[k]) 
                  -np.dot(self.SY[j],self.SY[k]) 
                  +2.0*np.dot(self.SZ[j],self.SZ[k])) 
          ham0 = ham0 + (dval/3.0)*dmat 
        # rhombic eeD w/o rotation [eeD,eeE/eeD]; e.g. [3.0,0.3333] = [-2,0,2] 
        elif len(eeDlist[i])==2: 
          dval = float(eeDlist[i][0]) 
          eod = float(eeDlist[i][1]) 
          dmat = dval*np.array([-1/3+eod,-1/3-eod,2/3]) 
          ham0 = ham0 + (dmat[0]*np.dot(self.SX[i],self.SX[i]) 
                        +dmat[1]*np.dot(self.SY[i],self.SY[i]) 
                        +dmat[2]*np.dot(self.SZ[i],self.SZ[i])) 
        # rhombic + rotate by Euler angles [D,E/D,alpha,beta,gamma] 
        # or full 3x3 matrix: [Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz] 
        elif len(eeDlist[i])==5 or len(eeDlist[i])==9: 
          if len(eeDlist[i])==5: 
            dmat = get_d_mat(eeDlist[i]) 
          else: 
            dmat = np.array(eeDlist[i]).reshape((3,3)) 
          for j in range(3): 
            for k in range(3): 
              ham0 = (ham0 
                     + dmat[j][k]*np.dot(self.S[j][i],self.S[k][i]))
        # unrecognized format 
        else: 
          raise_error('Uninterpretable eeD tensor: '+str(eeDlist[i])) 
    # make a 'Hamiltonian' object with these matrices 
    return Hamiltonian(ham0,hamx,hamy,hamz) 
  
  def spin_expectation(self, state): 
    # return spin expectation value at each center for a given state 
    expect = [] 
    for n in range(self.num): 
      expectx = np.dot(np.dot(np.conjugate(state),self.SX[n]),state) 
      expecty = np.dot(np.dot(np.conjugate(state),self.SY[n]),state) 
      expectz = np.dot(np.dot(np.conjugate(state),self.SZ[n]),state) 
      expect.append(np.array([expectx,expecty,expectz])/(-self.spins[n])) 
    return np.real(np.array(expect)) 

""" integrate: integrates spin projection vectors over a grid  
    - requires temperature and field as scalars, and a SpinSystem variable 
    - can take a variable list 
    - returns [[lxSx, lySy, lzSz] for each spin] """
def integrate(temperature,field, spinsystem, vars=None): 
  sum = np.zeros((spinsystem.num, 3)) 
  ham = spinsystem.ham(vars) 
  for pt in spinsystem.grid.grid: 
    contrib = np.zeros((spinsystem.num, 3)) 
    eval, evec = eig(ham.at_field(field*np.array(pt[:3]))) 
    eval = np.real(eval) # Hermitian so any imag part is numerical noise 
    eval = eval - np.min(eval) # shift lowest level to 0 to prevent overflows 
    eval = np.exp((boltzFactor/temperature)*eval) # exp(-E/(kB*T)) 
    eval = eval/np.sum(eval) # exp(-E/(kB*T)) / sum_k[exp(-E_k/(kB*T))] 
    for i in range(len(eval)): 
      if eval[i] == 0.0: # if pop underflows (not populated), skip eigenstate 
        continue 
      contrib += eval[i]*spinsystem.spin_expectation(evec[:,i]) 
    contrib = [pt[:3]*u for u in contrib] 
    sum += pt[3]*np.array(contrib) 
  return sum 

""" integrate_matrix: integrates 3x3 matrix <S>*<l> over a grid  
    - requires temperature and field as scalars, and a SpinSystem variable 
    - can take a variable list 
    - returns [[Sxlx, Sxly, Sxlz, Sylx, Syly, Sylz, Szlx, Szly, Szlz] for each
      spin center] 
    - useful for non-orthorhombic systems """
def integrate_matrix(temperature, field, spinsystem, vars=None): 
  sum = np.zeros((spinsystem.num, 9)) 
  ham = spinsystem.ham(vars) 
  for pt in spinsystem.grid.grid: 
    contrib = np.zeros((spinsystem.num, 3)) 
    eval, evec = eig(ham.at_field(field*np.array(pt[:3]))) 
    eval = np.real(eval) # Hermitian so any imag part is numerical noise 
    eval = eval - np.min(eval) # shift lowest level to 0 to prevent overflows 
    eval = np.exp((boltzFactor/temperature)*eval) # exp(-E/(kB*T)) 
    eval = eval/np.sum(eval) # exp(-E/(kB*T)) / sum_k[exp(-E_k/(kB*T))] 
    for i in range(len(eval)): 
      if eval[i] == 0.0: # if population underflow, don't bother 
        continue 
      contrib += eval[i]*spinsystem.spin_expectation(evec[:,i]) 
    contrib = [[u*v for u in w for v in pt[:3]] for w in contrib] 
    sum += pt[3]*np.array(contrib) 
  return sum 

""" reduce_pols_by_svd: 
    - takes polarizations as a np.array 
    - rhombicity can express one of two things: 
      - if rhombicity > 0 this is taken as a user-selected dimension 
      - if rhombicity < 0 its absolute value is the collinearity cutoff 
    - returns orthogonal composite polarizations and doesn't bother to 
      reorganize or scale composite polarizations so hopefully slightly faster 
      during fitting """ 
def reduce_pols_by_svd(polarizations, rhombicity): 
  U, s, Vh = svd(polarizations, full_matrices=False) 
  if rhombicity < 0: 
    s = s**2 
    s = s/np.sum(s) 
    rhombicity = sum(i > abs(rhombicity) for i in s) 
  rhombicity = max(rhombicity, 1) 
  return Vh[:rhombicity] 

""" reduce_pols_by_svd_full: 
    - takes polarizations as a np.array 
    - rhombicity can express one of two things: 
      - if rhombicity > 0 this is taken as a user-selected dimension
      - if rhombicity < 0 its absolute value is the collinearity cutoff 
    - returns [new_polarizations, conversion_matrix, normalized sing vals] """ 
def reduce_pols_by_svd_full(polarizations, rhombicity): 
  def transform_mat(mat): 
    # returns a transformation matrix between the unitary SVD transformations 
    # and something easier to interpret by the end user 
    # 0. append an identity matrix to keep track of operations 
    dim = len(mat[0]) 
    mat = np.append(mat, np.eye(dim), axis=0).T 
    # 1. reduce matrix (akin to red row echelon) 
    for i in range(dim): 
      pos = np.argmax(abs(mat[i,:-dim])) 
      for j in range(dim): 
        if i == j: 
          mat[j] = (mat[j,pos]/abs(mat[j,pos]))*mat[j] # make positive 
        else: 
          mat[j] = mat[j]-(mat[j,pos]/mat[i,pos])*mat[i] 
    # 2. normalize rows (just to standardize things) 
    for i in range(dim): 
      norm = np.sqrt(np.sum(mat[i,:-dim]**2)) 
      mat[i] = mat[i]/norm 
    # 3. rearrange (to bring closer to [Myz,Mzx,Mxy] order) 
    maxpos = [np.argmax(abs(col)) for col in mat.T[:-dim]] 
    rearr = [] 
    for x in maxpos: 
      if x not in rearr: rearr.append(x) 
    for x in range(dim): 
      if x not in rearr: rearr.append(x) 
    mat = mat[rearr] 
    # 4. return the transformation matrix (previously the identity matrix) 
    return mat.T[-dim:] 
  
  U, s, Vh = svd(polarizations, full_matrices=False) 
  Vh = np.dot(np.diag(s), Vh) 
  s = s**2 
  s = s/np.sum(s) 
  if rhombicity < 0: 
    rhombicity = sum(i > abs(rhombicity) for i in s) 
  rhombicity = max(rhombicity, 1) 
  U = U[:,:rhombicity]; Vh = Vh[:rhombicity] 
  t = transform_mat(U) 
  U = np.dot(t.T, U.T) 
  Vh = np.dot(inv(t), Vh) 
  return [Vh, U, s] 

""" Polarization class: """
class Polarization: 
  def __init__(self, pol=[], constraint=[], center=0): 
    self.center = center 
    self.rss = None # residual sum of square error
    self.wrss = None # weighted residual sum of square error 
    self.reg = pol # Cartesian polarizations 
    self.constraint = constraint # constraints 
    self.svd = [] # SVD polarizations 
    self.nsing = [] # normalized singular values 
    self.transform = [] # transformation matrix 
    self.covariance = [] 

""" VTVHDataSet class: """
class VTVHDataSet: 
  def __init__(self, datafile=None, runtype=None): 
    self.infile = datafile 
    self.outfile = '' 
    self.temp_field = [] # [[Temp1,Field1], [Temp2,Field2], [Temp3,Field3],...] 
    self.num = 0 # number of transitions within datafile 
    self.data = [] # [[y1,y2,y3...] for trans 1, [y1,y2,y3] for trans 2, ...] 
    self.error = [] # [[z1,z2,z3...] for trans 1, [z1,z2,z3] for trans 2, ...] 
    self.sim = [] # [[s1,s2,s3...] for trans 1, [s1,s2,s3] for trans 2, ...] 
    self.polarizations = []  # [[Transition1], [Transition2], ...] 
    if datafile is not None: 
      try: 
        self.read_datafile(datafile, runtype) 
      except FileNotFoundError: 
        raise_error('Data file not found', str(datafile)) 
      except OSError: 
        raise_error('Error reading file', str(datafile)) 
  
  def read_datafile(self, datafile, runtype, encoding='utf-8'): 
    self.infile = datafile 
    def is_number(x): 
      if isinstance(x, numeric): 
        return True 
      elif isinstance(x, str): # if it's a string, try to convert to a number 
        try: 
          return isinstance(float(x), float) 
        except ValueError: 
          return False 
    with open(datafile, newline='', encoding=encoding) as csv_file: 
      csv_reader = csv.reader(csv_file, delimiter=',') 
      for row in csv_reader: 
        if not is_number(row[0]): 
          continue # nonnumeric entry in row[0] is interpreted as a header row 
        row = np.array(row).astype(float) 
        if runtype == 'fit': 
          if len(row) >= 4 and len(row)%2 == 0: 
            self.temp_field.append(row[:2]) # first two entries are [T,H] 
            self.data.append(row[2::2]) # every other entry is data
            self.error.append(row[3::2]) # every other entry is error 
          else: 
            raise_error('Improperly formatted row in data file', datafile, 
                        '\n       ['+', '.join(row)+']',
                        '\n       Found',str(len(row)), 
                        'column(s), but expected an even number of at least 4',
                        '\n       formatted as [Temp, Field, Data1, Error1,', 
                                 'Data2, Error2, ...]') 
        else: # 'simulation' so only grab [T,H] information 
          if len(row)<2: 
            raise_error('Improperly formatted row in data file', datafile, 
                        '\n       ['+', '.join(row)+']',
                        '\n       Rows must have at least two columns:', 
                                 '[Temp, Field]') 
          self.temp_field.append(row[:2]) 
    self.temp_field = np.array(self.temp_field).astype(float) 
    self.data = np.transpose(self.data).astype(float) 
    self.error = np.transpose(self.error).astype(float) 
    self.num = len(self.data) 
  
  def write_datafile(self, runtype): 
    header = ['Temp', 'Field', 'bH/2kT'] 
    for i in range(self.num): 
      header.append('Sim '+str(i+1)) 
      header.append('Data '+str(i+1)) 
      header.append('Weight Dev '+str(i+1)) 
    output = self.temp_field.T 
    output = np.append(output, [htFactor*output[1]/output[0]], axis=0) 
    for i in range(self.num): 
      output = np.append(output, np.array([self.sim[i]]), axis=0) 
      if runtype == 'fit': 
        output = np.append(output, [self.data[i]], axis=0) 
        output = np.append(output, 
                           [(self.sim[i]-self.data[i])/self.error[i]], axis=0) 
      else: 
        output = np.append(output, [np.zeros(len(self.sim[i]))], axis=0) 
        output = np.append(output, [np.zeros(len(self.sim[i]))], axis=0) 
    output = np.append([header], output.T, axis=0) 
    with open(self.outfile, mode='w') as output_file: 
      writer = csv.writer(output_file, delimiter=',') 
      for r in output: 
        writer.writerow(r) 
    
  def get_pol_vectors(self, spinsystem, vars=None): 
    directions = [integrate(temp, field, spinsystem, vars=vars) 
                  for temp, field in self.temp_field] 
    return np.transpose(np.array(directions), (1,2,0)) 
  
  def simulate(self, spinsystem, vars=None): 
    pol_vectors = self.get_pol_vectors(spinsystem, vars=vars) 
    sim = [] 
    for i in range(self.num): 
      pol = self.polarizations[i] 
      if len(pol.reg) != 3: 
        raise_error('Improper polarization during simulation') 
      vector = pol_vectors[pol.center]
      sim.append(np.dot(pol.reg, vector)) 
    return np.array(sim) 
  
  def goodness_of_fit(self, spinsystem, vars=None, rhombicity=3): 
    def func(x, *args): 
      return np.dot(x, np.asarray(args)) 
    pol_vectors = self.get_pol_vectors(spinsystem, vars=vars) 
    red_vectors = np.array([reduce_pols_by_svd(vectors, rhombicity) 
                            for vectors in pol_vectors]) 
    cost = 0 
    for i in range(self.num): 
      pol = self.polarizations[i] 
      if len(pol.constraint) == 0: 
        xdata = np.transpose(red_vectors[pol.center]) 
      elif len(pol.constraint) == 3: 
        xdata = np.dot([pol.constraint], pol_vectors[pol.center]).T 
      else: 
        raise_error('Improper constraint', str(pol.constraint)) 
      # initial guess unit polariz 
      p0 = np.ones(len(xdata[0])).astype(float) 
      ydata = self.data[i] 
      sigma = self.error[i] 
      popt, pcov = curve_fit(func, xdata, ydata, p0, 
                             sigma=sigma, absolute_sigma=True) 
      wrss = func(xdata, *popt) 
      wrss = (ydata-wrss)/sigma 
      cost += np.sum(wrss**2) 
    return cost 
  
  def update_by_svd(self, spinsystem, vars=None, rhombicity=3): 
    def func(x, *args): 
      return np.dot(x, np.asarray(args)) 
    pol_vectors = self.get_pol_vectors(spinsystem, vars=vars) 
    red_vectors = [reduce_pols_by_svd_full(vectors, rhombicity) 
                   for vectors in pol_vectors] 
    for i in range(self.num): 
      pol = self.polarizations[i] 
      nsing = red_vectors[pol.center][2] # normalized singular values 
      if len(pol.constraint) == 0: 
        conv = red_vectors[pol.center][1]
        xdata = np.transpose(red_vectors[pol.center][0]) 
      elif len(pol.constraint) == 3: 
        xdata = np.dot([pol.constraint], pol_vectors[pol.center]).T 
      else: 
        raise_error('Improper constraint', str(pol.constraint)) 
      ydata = self.data[i] 
      sigma = self.error[i] 
      p0 = np.ones(len(xdata[0])).astype(float) # initial guess unit polariz 
      popt, pcov = curve_fit(func, xdata, ydata, p0, 
                             sigma=sigma, absolute_sigma=True) 
      rss = (ydata-func(xdata, *popt)) 
      wrss = rss/sigma 
      rss = np.sum(rss**2) 
      wrss = np.sum(wrss**2) 
      pol.svd = popt 
      pol.nsing, pol.rss, pol.wrss = nsing, rss, wrss 
      if len(pol.constraint) == 0: 
        pol.covariance = pcov 
        pol.transform = conv 
        pol.reg = np.dot(pinv(conv), popt) 
      else: 
        pol.rss, pol.wrss = rss, wrss 
        pol.covariance = pcov 
        pol.transform = None 
        pol.reg = popt*pol.constraint 

""" fit_at_each_particle: calculates the cost at each particle position """
def fit_at_each_particle(vars_of_each_particle, spinsys, datasets, 
                         rhombicity, printall=False): 
  wrss = [] 
  for particle in vars_of_each_particle: 
    cost = 0 
    for dataset in datasets: 
      cost += dataset.goodness_of_fit(spinsys, vars=particle, 
                                      rhombicity=rhombicity) 
    if printall: 
      print('Particle at', str(particle), 'with WRSS =', str(cost), flush=True) 
    wrss.append(cost) 
  return wrss 

""" particle_swarm_fit: performs the particle swarm optimization and (if 
    requested) writes the history of particle movements to a file """
def particle_swarm_fit(spinsystem, list_of_datasets, settings): 
  options = {'c1': settings['swarm']['personal'], 
             'c2': settings['swarm']['social'], 
             'w': settings['swarm']['inertial']} 
  bounds = np.array(spinsystem.var_bounds) 
  bounds = [bounds[:,0],bounds[:,1]] 
  if settings['swarm']['parallel'] < 1: settings['swarm']['parallel'] = None 
  optimizer=ps.single.GlobalBestPSO(n_particles=settings['swarm']['particles'], 
                                    dimensions=len(spinsystem.var_names), 
                                    options=options, bounds=bounds) 
  cost, pos = optimizer.optimize(fit_at_each_particle, 
                                 iters=settings['swarm']['iterations'], 
                                 n_processes=settings['swarm']['parallel'], 
                                 spinsys=spinsystem, 
                                 datasets=list_of_datasets, 
                                 rhombicity=settings['rhombicity'], 
                                 printall=settings['swarm']['print']) 
  print('Particle swarm variable optimization finished') 
  if settings['swarm']['history'] is not None: 
    with open(settings['swarm']['history'], mode='w') as output_file: 
      writer = csv.writer(output_file, delimiter=',') 
      header1 = ['','']+[i+1 for i in range(settings['swarm']['particles']) 
                             for __ in range(len(spinsystem.var_names))] 
      writer.writerow(header1) 
      header2 = spinsystem.var_names*settings['swarm']['particles'] 
      header2 = ['Iter','Cost']+header2 
      writer.writerow(header2) 
      best_cost = np.inf 
      cost_history = optimizer.cost_history 
      pos_history = optimizer.pos_history 
      vel_history = optimizer.velocity_history 
      for i in range(len(cost_history)): 
        if cost_history[i] < best_cost: 
          best_cost = cost_history[i] 
          print('Iteration', str(i+1)+': Global best cost', str(best_cost)) 
        row = [i+1, cost_history[i]] 
        for j in range(len(pos_history[i])): 
          row += list(pos_history[i][j]) 
        writer.writerow(row) 
    print('Particle movement history printed to', 
          settings['swarm']['history'], '\n') 
  return [cost, pos] 

""" read_yaml: A giant function that interprets user input from the 
    YAML file. Returns [settings, spin_system, datasets]. These are: 
    - settings: a list of settings for the run 
    - spin_system: a SpinSystem object containing all requisite information 
      to construct the spin Hamiltonian on the fly 
    - datasets: a list of VTVHDataSet objects """
def read_yaml(filename): 
  settings=dict() # make an empty dict in which to keep user settings etc. 
  variables=[] # an empty list of variable names 
  def lower(x): # if x is a string make lowercase, else return unchanged 
    return x.lower() if isinstance(x,str) else x 
  def pairs(x): # returns number of unordered pairs among x elements 
    return x*(x-1)/2 
  def fetch_val(yaml, str, default=None): 
    try: 
      return yaml[str] 
    except KeyError: 
      return default 
  def check_shape(mat, shapes): 
    # a g tensor can have shapes [1,2,3,6,9] 
    # a D tensor can have shapes [0,1,2,5,9] 
    # a eeG vector can have shape [0,3] 
    # a eeD vector can have shapes [0,1,2,5,9] 
    if not isinstance(mat, list): 
      return False 
    if not len(mat) in shapes: 
      return False 
    for x in mat: 
      if not isinstance(x, numeric) and not isinstance(x, str): 
        return False 
    return True 
  def scan_params(x): # search for fit vars and add accumulate in `variables' 
    nonlocal variables 
    if isinstance(x, (int,float)): 
      return True 
    elif isinstance(x, str): 
      if x not in variables: variables.append(x) 
      return True 
    elif isinstance(x, list): 
      return all([scan_params(y) for y in x]) 
    else: 
      return False 
  def index_vars(x): # switches to proper numeric indices from variable names 
    nonlocal variables 
    if isinstance(x, (int,float)): 
      return None 
    elif isinstance(x, str): 
      try: 
        return variables.index(x) 
      except ValueError: 
        raise_error('Error indexing variables:', str(x), 
                    'not in', str(variables)) 
    elif isinstance(x, (list, np.ndarray)): 
      return [index_vars(y) for y in x] 
    else: 
      raise_error('Error indexing variables: uninterpretable', str(x)) 
  
  try: 
    file = open(filename, 'r') 
    cfg = safe_load(file) 
    file.close() 
  except FileNotFoundError: 
    raise_error('Input file not found', filename) 
  except OSError as exc: 
    raise_error('Error opening file', filename, '\n', str(exc)) 
  except YAMLError as exc: 
    file.close() 
    raise_error('Error parsing YAML input file', filename, '\n', str(exc)) 
  
  settings['type'] = lower(fetch_val(cfg, 'type')) 
  if settings['type'] in ['sim', 'simulate', 'simulating', 'simulation']: 
    settings['type'] = 'simulation' 
  elif settings['type'] in ['fit', 'fitting']: 
    settings['type'] = 'fit' 
  elif settings['type'] is None: 
    raise_error('No run type specified') 
  else: 
    raise_error('Unrecognized run type requested: '+str(settings['type'])) 
  print('Run type:', settings['type']) 
  
  print('Reading data file(s)...') 
  settings['data'] = fetch_val(cfg, 'data') 
  if settings['data'] is None: 
    raise_error('No data file specified') 
  if isinstance(settings['data'], dict): 
    raise_error('Each data file must begin with a hyphenated (-) line')
  if not isinstance(settings['data'], list): 
    raise_error('Uninterpretable data file specification') 
  if not all([isinstance(x, dict) for x in settings['data']]): 
    raise_error('Incorrectly formatted (non-dict) list of data files') 
  datasets = [] 
  for i in range(len(settings['data'])): 
    if 'file' not in settings['data'][i]: 
      raise_error('File name missing for data file entry number', str(i+1)) 
    if not isinstance(settings['data'][i]['file'], str): 
      raise_error('Non-string \'file\' name found:', 
                  str(settings['data'][i]['file'])) 
    if 'out' not in settings['data'][i]: 
      settings['data'][i]['out'] = settings['data'][i]['file']+'.out.csv' 
    if not isinstance(settings['data'][i]['out'], str): 
      raise_error('Non-string \'out\' file name found:', 
                  str(settings['data'][i]['out'])) 
    print('- Reading document',str(i+1)+':',settings['data'][i]['file']) 
    x = VTVHDataSet(settings['data'][i]['file'], settings['type']) 
    x.outfile = settings['data'][i]['out'] 
    if settings['type'] == 'simulation': 
      # simulations require polarizations and accept center specifications 
      if 'polarizations' not in settings['data'][i]: 
        raise_error('Simulations require polarization info for each data set') 
      if not isinstance(settings['data'][i]['polarizations'], list): 
        raise_error('Polarizations must be in list form') 
      # check if we've been given a single polarization rather than list of one 
      if all([isinstance(x, numeric) 
              for x in settings['data'][i]['polarizations']]): 
        settings['data'][i]['polarizations'] = \
          [settings['data'][i]['polarizations']] 
      if 'centers' not in settings['data'][i]: 
        settings['data'][i]['centers'] = \
          [1]*len(settings['data'][i]['polarizations'])
      # check if we've been given a single center rather than list of one 
      if isinstance(settings['data'][i]['centers'], int): 
        settings['data'][i]['centers'] = [settings['data'][i]['centers']] 
      settings['data'][i]['centers'] = [k-1 for k 
                                        in settings['data'][i]['centers']] 
      if (len(settings['data'][i]['centers']) != 
          len(settings['data'][i]['polarizations'])): 
        raise_error('Number of centers specified different', 
                    'from number of polarizations') 
      for j in range(len(settings['data'][i]['polarizations'])): 
        if len(settings['data'][i]['polarizations'][j]) != 3: 
          print('  - Transition',str(j+1)) 
          raise_error('Polarizations for simulation must be 3-membered lists') 
        if not isinstance(settings['data'][i]['centers'][j], int): 
          print('  - Transition',str(j+1)) 
          raise_error('Centers must be integers, but', 
                      str(settings['data'][i]['centers'][j]), 
                      'is a', str(type(settings['data'][i]['centers'][j]))) 
        print('  - Transition', str(j+1), 'on center', 
              str(settings['data'][i]['centers'][j]+1)) 
        x.polarizations.append(
                Polarization(pol=settings['data'][i]['polarizations'][j], 
                center=settings['data'][i]['centers'][j])) 
    else: # 'fit' 
      # fits don't require polarizations, but if specified can only be (0/3) 
      # fits also accept center specifications 
      if 'polarizations' not in settings['data'][i]: 
        settings['data'][i]['polarizations'] = [[]]*x.num 
      if 'centers' not in settings['data'][i]: 
        settings['data'][i]['centers'] = [1]*x.num 
      # check if we have a single polarization rather than list of one 
      if all([isinstance(x, (float,int)) 
             for x in settings['data'][i]['polarizations']]): 
        settings['data'][i]['polarizations'] = \
          [settings['data'][i]['polarizations']] 
      # check if we have a single center rather than list of one 
      if isinstance(settings['data'][i]['centers'], int): 
        settings['data'][i]['centers'] = [settings['data'][i]['centers']] 
      settings['data'][i]['centers'] = [k-1 for k 
                                        in settings['data'][i]['centers']] 
      if len(settings['data'][i]['polarizations']) != x.num: 
        raise_error('Number of polarizations specified different', 
                    'from number of transitions in file') 
      if len(settings['data'][i]['centers']) != x.num: 
        raise_error('Number of centers specified different', 
                    'from number of transitions in file') 
      for j in range(x.num): 
        if len(settings['data'][i]['polarizations'][j]) not in [0,3]: 
          print('  - Transition',str(j+1)) 
          raise_error('Polarization constraints for fitting must be', 
                      '0- or 3-membered lists') 
        if not isinstance(settings['data'][i]['centers'][j], int): 
          print('  - Transition',str(j+1)) 
          raise_error('Centers must be integers, but', 
                      str(settings['data'][i]['centers'][j]), 
                      'is a', type(settings['data'][i]['centers'][j])) 
        pol = settings['data'][i]['polarizations'][j] 
        if np.sum(np.array(pol)**2) < 1e-6: # if constraint = zero vector 
          pol = [] 
        center = settings['data'][i]['centers'][j] 
        print('  - Transition', str(j+1), 'on center', str(center+1)) 
        x.polarizations.append(Polarization(constraint=pol, center=center)) 
    x.num = len(x.polarizations) 
    datasets.append(x) 
  print('Looking for output file name(s)...') 
  
  print('\nReading spins...') 
  spinsystem = fetch_val(cfg, 'spinsystem') 
  if spinsystem is None: 
    raise_error('No \'spinsystem\' list found') 
  if not isinstance(spinsystem, list): 
    raise_error('Check hyphenation of \'spinsystem\' list; none found') 
  if not all([isinstance(x, dict) for x in spinsystem]): 
    raise_error('Trouble interpreting \'spinsystem\' list') 
  try: 
    spins = [x['S'] for x in spinsystem] 
  except KeyError: 
    raise_error('Check hyphenation of \'spinsystem\' list', 
          '\n       Each hyphenated block must contain a spin \'S\'') 
  for i in range(len(spins)): 
    if not isinstance(spins[i], (int,float)): 
      raise_error('Improper spin:',str(spins[i])) 
    if spins[i] > 0: 
      spins[i] = int(round(2*spins[i]))/2 # round to nearest half integer 
    else: 
      raise_error('Diamagnet or improper spin:',str(spins[i])) 
  print('Constructing spin operator matrices...') 
  spin_system = SpinSystem(spins) 
  print('Parsing g tensors...') 
  gvalues = [] 
  for x in spinsystem: 
    try: 
      val = x['g'] 
    except KeyError: 
      val = [2.0023] 
    if isinstance(val, numeric) or isinstance(val, str): 
      val = [val] 
    if not isinstance(val, list): 
      raise_error('Non-list error for g tensor', str(val)) 
    if not check_shape(val, [1,2,3,6,9]): 
      raise_error('Uninterpretable g tensor', str(val)) 
    gvalues.append(val) 
  print('Parsing D (ZFS) tensors...') 
  Dvalues = [] 
  for x in spinsystem: 
    try: 
      val = x['D'] 
    except KeyError: 
      val = [] 
    if isinstance(val, numeric) or isinstance(val, str): 
      val = [val] 
    if not isinstance(val, list): 
      raise_error('Non-list error for D tensor', str(val)) 
    if not check_shape(val, [0,1,2,5,9]): 
      raise_error('Uninterpretable D tensor', str(val)) 
    Dvalues.append(val) 
  if spin_system.num == 1: 
    print('Single spin, so skipped exchange coupling') 
    Jvalues, eeGvalues, eeDvalues, Jfactor = [[], [], [], 1.0] 
  else: 
    print('Reading superexchange parameters...') 
    exchange = fetch_val(cfg, 'exchange') 
    if isinstance(exchange, numeric) or isinstance(exchange, str): 
      Jvalues = [exchange] 
    if isinstance(exchange, list): 
      Jvalues = exchange 
      eeGvalues = eeDvalues = [] 
      Jfactor = 1.0 
    elif isinstance(exchange, dict): 
      Jvalues = fetch_val(exchange, 'scalar', []) 
      eeGvalues = fetch_val(exchange, 'vector', []) 
      eeDvalues = fetch_val(exchange, 'tensor', []) 
      Jfactor = fetch_val(exchange, 'factor', 1.0) 
    if not isinstance(Jfactor, numeric): 
      raise_error('Inappropriate Jfactor value of', str(Jfactor)) 
    else: 
      terms = {1.0: '+J', 2.0: '+2J', -1.0: '-J', -2.0: '-2J'} 
      try: 
        x = terms[round(Jfactor,1)] 
      except KeyError: 
        x = str(Jfactor) if Jfactor<0 else '+'+str(Jfactor) 
      print('- Using',x,'term in Hamiltonian') 
    if not isinstance(Jvalues, list): Jvalues = [Jvalues] 
    if not isinstance(eeGvalues, list): eeGvalues = [eeGvalues] 
    if not isinstance(eeDvalues, list): eeDvalues = [eeDvalues] 
    if all([isinstance(x, numeric) or isinstance(x, str) for x in eeGvalues]): 
      if len(eeGvalues) > 0: 
        eeGvalues = [eeGvalues] 
    if all([isinstance(x, numeric) or isinstance(x, str) for x in eeDvalues]): 
      if len(eeDvalues) > 0: 
        eeDvalues = [eeDvalues] 
    if len(Jvalues)>0: 
      if not check_shape(Jvalues, [pairs(spin_system.num)]): 
        print('Improper Jvalues list of',str(Jvalues)) 
        Jvalues = []
        for x in range(spin_system.num): 
          for y in range(x+1,spin_system.num): 
            Jvalues.append('J'+str(x+1)+str(y+1)) 
        Jvalues = ','.join(Jvalues) 
        raise_error('Expected list of form ['+Jvalues+']') 
      print('- Found scalar (isotropic) J coupling') 
    else: 
      print('- No scalar (isotropic) J coupling found') 
    if len(eeGvalues)!=0 and len(eeGvalues)!=pairs(spin_system.num): 
      print('Improper number of entries in vector list:',str(eeGvalues)) 
      eeGvalues = []
      for x in range(spin_system.num): 
        for y in range(x+1,spin_system.num): 
          eeGvalues.append('G'+str(x+1)+str(y+1)) 
      eeGvalues = ','.join(eeGvalues) 
      raise_error('Expected list of form ['+eeGvalues+']') 
    else: 
      for x in range(len(eeGvalues)): 
        if not isinstance(eeGvalues[x], list): 
          print('Vector entry',str(x+1),'is not a list:',str(eeGvalues[x])) 
          raise_error('Expected [r,theta,phi]') 
        if not check_shape(eeGvalues[x], [3]): 
          print('Vector entry',str(x+1),'uninterpretable:',str(eeGvalues[x])) 
          raise_error('Expected [r,theta,phi]') 
      if len(eeGvalues)==0: 
        print('- No vector (antisymmetric) G coupling found') 
      else: 
        print('- Found vector (antisymmetric) G coupling') 
    if len(eeDvalues)!=0 and len(eeDvalues)!=pairs(spin_system.num): 
      print('Improper number of entries in tensor list:',str(eeDvalues)) 
      eeDvalues = []
      for x in range(spin_system.num): 
        for y in range(x+1,spin_system.num): 
          eeDvalues.append('D'+str(x+1)+str(y+1)) 
      eeDvalues = ','.join(eeDvalues) 
      raise_error('Expected list of form ['+eeDvalues+']') 
    else: 
      for x in range(len(eeDvalues)): 
        if not isinstance(eeDvalues[x], list): 
          raise_error('Each tensor must be provided as a list, not', 
                      str(eeDvalues[x])) 
        if not check_shape(eeDvalues[x], [0,1,2,5,9]): 
          raise_error('Tensor entry',str(x+1),'uninterpretable:', 
                      str(eeGvalues[x])) 
      if len(eeDvalues)==0: 
        print('- No tensor (anisotropic) D coupling found') 
      else: 
        print('- Found tensor (anisotropic) D coupling') 
  if not all(map(scan_params,[gvalues,Dvalues,Jvalues,eeGvalues,eeDvalues])): 
    if not scan_params(gvalues): 
      print('Unexpected issue with g tensors') 
    if not scan_params(Dvalues): 
      print('Unexpected issue with D (ZFS) tensors') 
    if not scan_params(Jvalues): 
      print('Unexpected issue with J exchange scalars') 
    if not scan_params(eeGvalues): 
      print('Unexpected issue with G exchange vectors') 
    if not scan_params(eeDvalues): 
      print('Unexpected issue with D exchange tensors') 
    raise_error('Problem scanning variables') 
  gvars = index_vars(gvalues) 
  dvars = index_vars(Dvalues) 
  jvars = index_vars(Jvalues) 
  eeGvars = index_vars(eeGvalues) 
  eeDvars = index_vars(eeDvalues) 
  if len(variables) > 0: 
    if settings['type'] == 'simulation': 
      raise_error('No variables allowed in simulation run') 
    print('Locating bounds for magnetic variables...') 
  var_bounds = [] 
  for i in variables: 
    bound = fetch_val(cfg, i) 
    if bound is None: raise_error('No bounds found for variable',str(i)) 
    if not isinstance(bound, list): 
      raise_error('Bounds for variable "'+str(i)+'" not a two-element list') 
    if len(bound) != 2: 
      raise_error('Bounds for variable "'+str(i)+'" not a two-element list') 
    if all([isinstance(x, (int,float)) for x in bound]): 
      var_bounds.append(np.sort(np.array(bound))) 
    else: raise_error('Bounds for variable "'+str(i)+'" must be numeric') 
  spin_system.set_mag_values(gvalues, Dvalues, 
                             Jvalues, eeGvalues, eeDvalues, Jfactor) 
  spin_system.set_mag_variables(gvars, dvars, jvars, eeGvars, eeDvars) 
  spin_system.initiate_variables(variables, var_bounds) 
  rhombicity = {1: 'isotropic', 2: 'axial', 3: 'rhombic'} 
  # detect magnetic symmetry from g and D tensors 
  auto_rhomb = max(max([min(len(x), 3) for x in gvalues]), 
              max([min(len(x)+1,3) for x in Dvalues])) 
  if len(eeGvalues) > 0: 
    auto_rhomb = 3 
  if len(eeDvalues) > 0: 
    auto_rhomb = max(auto_rhomb, max([min(len(x)+1,3) for x in eeDvalues])) 
  print('This is likely a maximally',rhombicity[auto_rhomb],'system') 
  print('Magnetic parameters read without error\n') 
  
  print('Contructing integration grid...') 
  integration = fetch_val(cfg, 'integration', dict()) 
  if isinstance(integration, list): 
    raise_error('Improperly formatted \'integration\' block', 
                '\n       Please remove any hyphenation') 
  if not isinstance(integration, dict): 
    raise_error('Improperly formatted \'integration\' block', 
                '\n       Uninterpretable non-dict') 
  method = lower(fetch_val(integration, 'method', 'gaussian')) 
  grid = fetch_val(integration, 'precision', 0) 
  domain = lower(fetch_val(integration, 'domain', 'auto')) 
  intgrid = IntegrationGrid(method, grid, domain) 
  spin_system.set_grid(intgrid) 
  if settings['type'] == 'fit': 
    print('Reading symmetry block...') 
    symmetry = fetch_val(cfg, 'symmetry', dict()) 
    if isinstance(symmetry, list): 
      raise_error('Improperly formatted \'symmetry\' block', 
                  '\n       Please remove any hyphenation') 
    if not isinstance(symmetry, dict): 
      raise_error('Improperly formatted \'symmetry\' block', 
                  '\n       Uninterpretable non-dict') 
    settings['rhombicity'] = lower(fetch_val(symmetry, 'rhombicity', 'auto')) 
    rhomb = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 'dynamic': -1, 'auto': 0, 
             'isotropic': 1, 'axial': 2, 'rhombic': 3} 
    if settings['rhombicity'] in [-1, 0, 1, 2, 3, 'dynamic', 
                                  'auto', 'isotropic', 'axial', 'rhombic']: 
      settings['rhombicity'] = rhomb[settings['rhombicity']] 
    else: 
      raise_error('Unrecognized user-input rhombicity', settings['rhombicity']) 
    settings['collinearity'] = fetch_val(symmetry, 'collinearity', 6) 
    if not isinstance(settings['collinearity'], numeric): 
      raise_error('Non-numeric collinearity setting') 
    if settings['collinearity'] < 1: 
      raise_error('Collinearity settings must be greater than 1') 
    print('Reading particle swarm parameters for fitting...') 
    settings['swarm'] = fetch_val(cfg, 'swarm', dict()) 
    if isinstance(settings['swarm'], list): 
      raise_error('Improperly formatted \'swarm\' block', 
                  '\n       Please remove any hyphenation') 
    if not isinstance(settings['swarm'], dict): 
      raise_error('Improperly formatted \'swarm\' block', 
                  '\n       Uninterpretable non-dict') 
    settings['swarm']['personal']=  fetch_val(settings['swarm'],
                                              'personal', 2.5) 
    settings['swarm']['social']=    fetch_val(settings['swarm'],
                                              'social', 1.3) 
    settings['swarm']['inertial']=  fetch_val(settings['swarm'],
                                              'inertial', 0.40) 
    settings['swarm']['particles']= fetch_val(settings['swarm'],
                                              'particles', 20) 
    settings['swarm']['iterations']=fetch_val(settings['swarm'],
                                              'iterations', 400)
    settings['swarm']['parallel']=  fetch_val(settings['swarm'],
                                              'parallel', 0) 
    settings['swarm']['print']=     fetch_val(settings['swarm'],
                                              'print', False) 
    settings['swarm']['history']=   fetch_val(settings['swarm'], 'history') 
    if not isinstance(settings['swarm']['personal'], (int,float)): 
      raise_error('Improper particle swarm personal (cognitive, c1) parameter') 
    if not isinstance(settings['swarm']['social'], (int,float)): 
      raise_error('Improper particle swarm social (group, c2) parameter') 
    if not isinstance(settings['swarm']['inertial'], (int,float)): 
      raise_error('Improper particle swarm inertial (w) parameter') 
    if not isinstance(settings['swarm']['particles'], int): 
      raise_error('Improper particle swarm number of particles') 
    if not isinstance(settings['swarm']['iterations'], int): 
      raise_error('Improper particle swarm number of iterations') 
    if not isinstance(settings['swarm']['parallel'], int): 
      raise_error('Must specify an integer number of processors') 
    if settings['swarm']['parallel'] < 0: 
      raise_error('Must specify a positive number of processors') 
    if lower(settings['swarm']['print']) in [True,1,'true','yes']: 
      settings['swarm']['print'] = True 
    elif lower(settings['swarm']['print']) in [False,0,'false','no']: 
      settings['swarm']['print'] = False 
    else: raise_error('Improper particle swarm printing setting') 
    if settings['swarm']['history'] is not None: 
      if not isinstance(settings['swarm']['history'], str): 
        raise_error('Non-string particle swarm history destination') 
    print('Reading other fitting parameters...') 
    settings['statistics'] = lower(fetch_val(cfg, 'statistics', True))
    if settings['statistics'] in [True, 1, 'true', 'yes']: 
      settings['statistics'] = True 
    elif settings['statistics'] in [False, 0, 'false', 'no']: 
      settings['statistics'] = False 
    else: raise_error('Improper statistics flag') 
  
  print('\nSummary of job:') 
  print('- Run type:', settings['type']) 
  print('- Data file(s):') 
  for i in range(len(datasets)): 
    print('  -', settings['data'][i]['file'], 
               'with', str(datasets[i].num), 
               'transition(s) at', str(len(datasets[i].temp_field)), 'points') 
    print('    - Output written to', str(datasets[i].outfile)) 
    for j in range(datasets[i].num): 
      pol = datasets[i].polarizations[j] 
      if 0 <= pol.center < spin_system.num: 
        if settings['type'] == 'simulation': 
          print('    - Transition', str(j+1), 'on center', str(pol.center+1),
                      'polarized as') 
          if len(pol.reg) == 3: 
            print('      [Myz,Mzx,Mxy] =', str(pol.reg)) 
          else: raise_error('Inappropriate polarization for simulation') 
        else: # 'fit' 
          if len(pol.constraint) == 0: 
            print('    - Transition', str(j+1), 'on center', str(pol.center+1), 
                        'unconstrained') 
          elif len(pol.constraint) == 3: 
            print('    - Transition', str(j+1), 'on center', str(pol.center+1), 
                        'constrained parallel to') 
            print('      [Myz,Mzx,Mxy] =', str(pol.constraint)) 
          else: raise_error('Inappropriate polarization for fitting') 
      else: raise_error('Center out of range:', str(datasets[i].centers[j]+1)) 
  print('- Numerical integration') 
  if intgrid.method == 'gaussian': 
    print('  - Method: Gaussian quadrature') 
    print('  - Domain:', intgrid.domain) 
    print('  - Polar degree of', intgrid.deg[0]) 
    print('  - Azimuthal degree of', intgrid.deg[1]) 
    print('  - Overall grid of', intgrid.size, 'points') 
  elif intgrid.method == 'lebedev': 
    print('  - Method: Lebedev quadrature') 
    print('  - Domain:', intgrid.domain) 
    print('  - Precision degree of', intgrid.deg) 
    print('  - Overall grid of', intgrid.size, 'points') 
  elif intgrid.method == 'discrete': 
    print('  - Method: Discrete (user-defined)') 
    print('  - Domain specified by user') 
    print('  - Overall grid of', intgrid.size, 'points') 
  if settings['type'] == 'fit': 
    if len(spin_system.var_names) > 0: 
      print('- Particle swarm fitting parameters') 
      print('  - Personal/cognitive (c1) parameter:', 
                 str(settings['swarm']['personal'])) 
      print('  - Social/group (c2) parameter:', 
                 str(settings['swarm']['social'])) 
      print('  - Inertial (w) parameter:', 
                 str(settings['swarm']['inertial'])) 
      print('  -',str(settings['swarm']['particles']),'particles') 
      print('  -',str(settings['swarm']['iterations']),'iterations') 
      if settings['swarm']['parallel'] > 1: 
        print('  -',str(settings['swarm']['parallel']),'parallel processes') 
      else:
        print('  - Serial execution') 
      if settings['swarm']['print']: 
        print('  - Full (unprocessed) particle position printing requested') 
      if settings['swarm']['history'] is not None: 
        print('  - Particle position history to be written to', 
                   settings['swarm']['history']) 
    else: 
      print('- No variables require particle swarm fitting')
    print('- Other fitting parameters') 
    if settings['statistics']: 
      print('  - Statistical analysis enabled') 
    else: print('  - Statistics module disabled') 
    if settings['rhombicity'] == 0: 
      print('  - Automatic symmetry detection used') 
      settings['rhombicity'] = auto_rhomb 
    elif settings['rhombicity'] == -1: 
      print('  - Dynamically updated rhombicity') 
      settings['rhombicity'] = -10.0**(-settings['collinearity']) 
    elif settings['rhombicity'] == 1: 
      print('  - User-specified isotropic symmetry enforced') 
    elif settings['rhombicity'] == 2: 
      print('  - User-specified axial symmetry enforced') 
    else: #settings['rhombicity'] == 3: 
      print('  - User-specified rhombic symmetry enforced') 
    print('  - Collinearity cutoff of', str(settings['collinearity']), 
              '('+str(10.0**(-settings['collinearity']))+')') 
  print('- Spin Hamiltonian parameters') 
  for i in range(spin_system.num): 
    print('  - Spin center', str(i+1)+',', 'S =', str(spin_system.spins[i])) 
    print('    g =', str(spin_system.g_vals[i])) 
    print('    D =', str(spin_system.d_vals[i])) 
  if spin_system.num > 1: 
    if len(spin_system.j_vals) == 0: print('  - Scalar J coupling: None') 
    else: 
      print('  - Scalar J coupling:') 
      for i in range(len(spin_system.j_vals)): 
        x = 'J' +str(spin_system.jindex[i,0]+1) +str(spin_system.jindex[i,1]+1) 
        print('    '+x, '=', str(spin_system.j_vals[i])) 
    if len(spin_system.eeG_vals) == 0: 
      print('  - Vector eeG coupling: None') 
    else: 
      print('  - Vector eeG coupling:') 
      for i in range(len(spin_system.eeG_vals)): 
        x = 'G' +str(spin_system.jindex[i,0]+1) +str(spin_system.jindex[i,1]+1) 
        print('    '+x+'(r,theta,phi) =', str(spin_system.eeG_vals[i])) 
      print('    WARNING: Verify integration domain is acceptible!') 
    if len(spin_system.eeD_vals) == 0: 
      print('  - Tensor eeD coupling: None') 
    else: 
      print('  - Tensor eeD coupling:') 
      for i in range(len(spin_system.eeD_vals)): 
        x = 'D' +str(spin_system.jindex[i,0]+1) +str(spin_system.jindex[i,1]+1) 
        print('    '+x, '=', str(spin_system.eeD_vals[i])) 
      print('    WARNING: Verify integration domain is acceptible!') 
  if len(spin_system.var_names) > 0: 
    print('- Fitting variables:') 
    for i in range(len(spin_system.var_names)): 
      print('  -',spin_system.var_names[i],'in', 
            '['+', '.join([str(x) for x in spin_system.var_bounds[i]])+']') 
  return [settings, spin_system, datasets] 

""" raw_summary: summarize results from fit without statistical analysis """
def raw_summary(spinsystem, datasets): 
  print('\nSummary of fit, unprocessed:') 
  if len(spinsystem.var_names) == 0: 
    print('- No fitted variables') 
  else: 
    print('- Fitted variables:') 
    for i in range(len(spinsystem.var_names)): 
      print('  - Refined', spinsystem.var_names[i], 'to', 
            str(spinsystem.vars[i])) 
  x = [pol.wrss for dataset in datasets for pol in dataset.polarizations] 
  x = np.sum(x) 
  print('- Weighted residual sum of squares:', str(x)) 
  for i in range(len(datasets)): 
    for j in range(datasets[i].num): 
      print('  - Dataset', str(i+1), 'transition', str(j+1), 
            'with WRSS =', str(datasets[i].polarizations[j].wrss)) 
  print('- Spin Hamiltonian parameters:') 
  glist   = resolve_variables(spinsystem.g_vals, spinsystem.g_vars, 
                              spinsystem.vars) 
  dlist   = resolve_variables(spinsystem.d_vals, spinsystem.d_vars, 
                              spinsystem.vars) 
  jlist   = resolve_variables(spinsystem.j_vals, spinsystem.j_vars, 
                              spinsystem.vars) 
  eeGlist = resolve_variables(spinsystem.eeG_vals, spinsystem.eeG_vars, 
                              spinsystem.vars) 
  eeDlist = resolve_variables(spinsystem.eeD_vals, spinsystem.eeD_vars, 
                              spinsystem.vars) 
  for i in range(spinsystem.num): 
    print('  - Spin number', str(i+1)+',', 'S =', str(spinsystem.spins[i])) 
    print('    g =', sf(glist[i])) 
    print('    D =', sf(dlist[i])) 
  if spin_system.num > 1: 
    if len(jlist) == 0: print('  - Scalar J exchange coupling: None') 
    else: 
      print('  - Scalar J exchange coupling:') 
      for i in range(len(jlist)): 
        print('    J'+str(spin_system.jindex[i,0]+1)+str(spin_system.jindex[i,1]+1), 
              '=', sf(jlist[i])) 
    if len(eeGlist) == 0: print('  - Vector G exchange coupling: None') 
    else: 
      print('  - Vector G exchange coupling:') 
      for i in range(len(eeGlist)): 
        print('    G'+str(spin_system.jindex[i,0]+1)+str(spin_system.jindex[i,1]+1) 
              +'(r,theta,phi) =', sf(eeGlist[i])) 
    if len(eeDlist) == 0: print('  - Tensor D exchange coupling: None') 
    else: 
      print('  - Tensor D exchange coupling:') 
      for i in range(len(eeDlist)): 
        print('    D'+str(spin_system.jindex[i,0]+1)+str(spin_system.jindex[i,1]+1), 
              '=', sf(eeDlist[i])) 
  else: 
    print('  - Single spin, so no exchange coupling') 
  print('- Transition polarizations:') 
  for i in range(len(datasets)): 
    for j in range(datasets[i].num): 
      pol = datasets[i].polarizations[j] 
      conv = pol.transform 
      if conv is None: 
        print('  - Dataset', str(i+1), 'constrained transition', str(j+1), 
              'polarized as') 
        if len(pol.reg) == 3: 
          print('      [Myz, Mzx, Mxy] =', sf(pol.reg)) 
        else: 
          raise_error('Improperly sized polarization,', str(pol.size)) 
      else: 
        # let's check if the transformation matrix is the identity matrix 
        is_identity_mat = False 
        if conv.shape[0] == conv.shape[1]: 
          is_identity_mat = conv - np.eye(conv.shape[0]) 
          is_identity_mat = np.dot(is_identity_mat, is_identity_mat.T) 
          if np.trace(is_identity_mat) < 1e-6: 
            is_identity_mat = True 
          else: 
            is_identity_mat = False 
        if is_identity_mat: 
          print('  - Dataset', str(i+1), 'transition', str(j+1), 
                'polarized as') 
          if len(pol.svd) == 3: 
            print('      [Myz, Mzx, Mxy] =', sf(pol.svd)) 
          else: 
            raise_error('Improperly sized polarization,', str(pol.size)) 
        else: 
          polstr = ['M'+str(k+1)+' '+sf(pol.svd[k]) 
                    for k in range(len(pol.svd))] 
          print('  - Dataset', str(i+1), 'transition', str(j+1), 
                'polarized as', ', '.join(polstr)) 
          print_matrix_eq(conv, three_pol_mat, '=', 
                          gen_pol_mat[:len(pol.svd)], indent=4) 

""" statistics_module: calculate (unweighted) covariance matrix and report 
    (hopefully) useful statistics information to the user """
def statistics_module(spinsystem, datasets): 
  def get_covariance(spinsystem, datasets): 
    if len(spinsystem.var_names) == 0: 
      return np.array([]) 
    epsilon = 0.01 # change if you'd like! 
    displacements = epsilon * np.eye(len(spinsystem.var_names)) 
    best_fit = np.array([x for dataset in datasets 
                           for transition in dataset.sim 
                           for x in transition]) # flat array of simulated vals 
    vars = spinsystem.vars
    jacobian = [[x for dataset in datasets 
                   for transition in dataset.simulate(spinsystem, vars + disp) 
                   for x in transition] for disp in displacements] 
    jacobian = (1/epsilon)*np.array([x - best_fit for x in jacobian]) 
    variance = np.sum([pol.rss for dataset in datasets 
                               for pol in dataset.polarizations]) 
    variance = variance / (len(best_fit)-len(spinsystem.var_names)) 
    covariance = np.dot(jacobian, jacobian.T) 
    covariance = variance * inv(covariance) 
    return covariance 
  
  print('\nStatistical Module:') 
  print('- Estimating error by numerical differentiation/Jacobian') 
  names = spinsystem.var_names 
  cov = get_covariance(spinsystem, datasets) 
  if len(names) == 0: 
    print('- No fitted variables') 
  else: 
    print('- Fitted variables:') 
    print('  - Covariances, unweighted:') 
    for v in range(len(names)): 
      print('    - sigma^2['+names[v]+'] =', sf(cov[v,v])) 
    for v in range(len(names)): 
      for w in range(v+1, len(names)): 
        print('    - sigma['+names[v]+','+names[w]+'] =', sf(cov[v,w])) 
    print('  - Correlation coefficients, unweighted:') 
    for v in range(len(names)): 
      for w in range(v+1, len(names)): 
        corr = cov[v,w]/np.sqrt(cov[v,v]*cov[w,w]) 
        print('    - corr['+names[v]+','+names[w]+'] =', sf(corr)) 
    print('  - Standard error of fit, unweighted:') 
    for v in range(len(names)): 
      num_err = cov[v,v] 
      if not np.isnan(num_err): 
        num_err = np.sqrt(num_err) 
      num_err = number_error(spinsystem.vars[v], num_err) 
      print('    - Variable', names[v], '=', num_err) 
  for d, dataset in enumerate(datasets): 
    print('- Polarizations for dataset', str(d+1), '('+dataset.infile+')') 
    for p, pol in enumerate(dataset.polarizations): 
      # let's check if the transformation matrix is the identity matrix 
      conv = pol.transform 
      cov = pol.covariance 
      print('  - Transition', str(p+1)+':') 
      if len(pol.nsing) > 0: 
        print('    - -Log10(Norm square sing vals):', 
            '\n     ', sf(-np.log10(pol.nsing), sig=2)) 
      if conv is None: 
        print('    - Constrained polarization') 
        print('    - Covariance for proportionality constant, weighted:', 
            '\n      sigma^2 =', str(cov[0,0])) 
        print('    - Standard error of the fit for proportionality constant,', 
              'weighted:\n      M =', number_error(pol.svd[0], 
              np.sqrt(cov[0,0]))) 
      else: 
        is_identity_mat = False 
        if conv.shape[0] == conv.shape[1]: 
          is_identity_mat = conv - np.eye(conv.shape[0]) 
          is_identity_mat = np.dot(is_identity_mat, is_identity_mat.T) 
          if np.trace(is_identity_mat) < 1e-6: 
            is_identity_mat = True 
          else: 
            is_identity_mat = False 
        if is_identity_mat: 
          pol_names = ['Myz', 'Mzx', 'Mxy'] 
        else: 
          pol_names = ['M1', 'M2', 'M3'][:len(pol.svd)] 
        print('    - Covariances, weighted:') 
        for v in range(len(pol.svd)): 
          print('      - sigma^2['+pol_names[v]+'] =', sf(cov[v,v])) 
        for v in range(len(pol.svd)): 
          for w in range(v+1,len(pol.svd)): 
            print('      - sigma['+pol_names[v]+',' 
                  +pol_names[w]+'] =', sf(cov[v,w])) 
        print('    - Correlation coefficients, weighted:') 
        for v in range(len(pol.svd)): 
          for w in range(v+1,len(pol.svd)): 
            corr = cov[v,w]/np.sqrt(cov[v,v]*cov[w,w]) 
            print('      - corr['+pol_names[v]+','+pol_names[w]+'] =', sf(corr)) 
        print('    - Standard error of the fit, weighted:') 
        for v in range(len(pol.svd)): 
          num_err = cov[v,v] 
          if not np.isnan(num_err): 
            num_err = np.sqrt(num_err) 
          num_err = number_error(pol.svd[v], num_err) 
          print('      '+pol_names[v], '=', num_err) 
  return True 

""" Here we go, the main program """ 
if __name__ == "__main__": 
  print('MCD Simulation and Fitting with Particle Swarm Optimization') 
  print('Version 0.5, Wesley J. Transue\n') 
  
  if len(sys.argv) != 2: 
    raise_error('Please provide one input YAML file to be processed') 
  
  start_time = time.time() 
  [settings, spin_system, datasets] = read_yaml(sys.argv[1]) 
  
  print('\nBeginning job...', flush=True)
  if settings['type'] == 'fit': 
    # if there are variables to be fit, do so then update spin_system 
    if len(spin_system.var_names) > 0: 
      cost, pos = particle_swarm_fit(spin_system, datasets, settings) 
      spin_system.set_vars(pos) 
    # fit polarizations for each data set, possibly with SVD reduction 
    for dataset in datasets: 
      dataset.update_by_svd(spin_system, rhombicity=settings['rhombicity']) 
  for dataset in datasets: 
    dataset.sim = dataset.simulate(spin_system) 
    dataset.write_datafile(settings['type']) 
  if settings['type'] == 'fit': 
    raw_summary(spin_system, datasets) 
    if settings['statistics']: 
      statistics_module(spin_system, datasets) 
  
  express_duration(start_time) 
