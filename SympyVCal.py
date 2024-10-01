# Imports
import sympy as sp
from sympy import sin, diff
from sympy.vector import CoordSys3D



# Setting up 3 coordinate systems
Ca = CoordSys3D('Ca')                                       # Cartesian
Cy = Ca.create_new('Cy', transformation='cylindrical')      # Cylindrical
S = Ca.create_new('S', transformation='spherical')          # Spherical




# Describing cylindrical unit vectors
rho, phi, z = sp.symbols('rho phi z')

# Describing spherical unit vectors (I initialized phi again to not confuse myself)
r, theta, phi = sp.symbols('r theta phi')



# Cylindrical Vector Calculus
def Cy_grad(fun):
   '''
   The gradient in cylindrical coordinates.

   For a scalar function 𝑓, the gradient in cylindrical coordinates is given by:
    𝝯𝑓 = (∂𝑓/∂𝜌)𝐫 + [1/𝜌(∂𝑓/∂𝜃)]𝛉 + (∂𝑓/∂𝑧)𝐳
   '''
   rho_term = diff(fun, rho)
   theta_term = 1/rho * diff(fun, theta)
   z_term = diff(fun, z)

   return rho_term * Cy.i + theta_term * Cy.j + z_term * Cy.k



def Cy_div(vec):
   '''
   The divergence in cylindrical coordinates.

   For a vector <𝐫, 𝛉, 𝐳>, the divergence in cylindrical coordinates is given by:
   𝝯⦁𝑓 = (1/𝜌)(∂/∂𝜌[𝜌*𝐫]) + (1/𝜌)(∂𝛉/∂𝜃) + (∂𝐳/∂𝑧)
   '''
   i = vec.dot(Cy.i)
   j = vec.dot(Cy.j)
   k = vec.dot(Cy.k)

   rho_term = 1/rho * diff(i * rho, rho)
   theta_term = 1/rho * diff(j, theta)
   z_term = diff(k, z)

   return rho_term + theta_term + z_term



def Cy_curl(vec):
   '''
   The curl in cylindrical coordinates.

   For a vector <𝐫, 𝛉, 𝐳>, the curl in cylindrical coordinates is given by:
   𝝯x𝑓 = (1/𝜌)[(∂𝐳/∂𝜃) - (∂𝛉/∂𝑧)]𝐫 + [(∂𝐫/∂𝑧) - (∂𝐳/∂𝜌)]𝛉 + (1/𝜌)[(∂/∂𝜌(𝜌𝛉)) - (∂𝐫/∂𝜃)]𝐳
   '''
   i = vec.dot(Cy.i)
   j = vec.dot(Cy.j)
   k = vec.dot(Cy.k)

   rho_term = 1/rho * (diff(k, theta) - diff(j, z))
   theta_term = diff(i, z) - diff(k, rho)
   z_term = 1/rho * (diff(rho * j) - diff(i, theta))

   return rho_term * Cy.i + theta_term * Cy.j + z_term * Cy.k



def Cy_lapl(fun):
   '''
   The Laplacian in cylindrical coordinates.

   For a scalar function 𝑓, the Laplacian in cylindrical coordinates is given by:
   𝝯²𝑓 = (1/𝜌)(∂/∂𝜌[𝜌(∂𝑓/∂𝜌)]) + (1/𝜌²)(∂²𝑓/∂𝜃²) + (∂²𝑓/∂𝑧²)
   '''
   rho_term = 1/rho * diff(rho*diff(fun, rho), rho)
   theta_term = 1/rho**2 * diff(diff(fun, theta), theta)
   z_term = diff(diff(fun, z), z)

   return rho_term + theta_term + z_term



# Spherical Vector Calculus
def S_grad(fun):
    '''
    The gradient in spherical coordinates.

    For a function 𝑓, the gradient in spherical coordinates is given by:
    𝝯𝑓 = (∂𝑓/∂𝑟)𝐫 + [1/𝑟(∂𝑓/∂𝜃)]𝛉 + [(1/𝑟sin(𝜃))(∂𝑓/∂𝜑)]𝛗
    '''
    radial_term = diff(fun, r)
    theta_term = 1/r * diff(fun, theta)
    phi_term = 1/(r*sin(theta)) * diff(fun, phi)

    return radial_term * S.i + theta_term * S.j + phi_term * S.k


def S_div(vec):
    '''
    The divergence in spherical coordinates.

    For a vector <𝐫, 𝛉, 𝛗>, the divergence in spherical coordinates is given by:
    𝝯⦁𝑓 = (1/𝑟²)(∂/∂𝑟[𝑟²*𝐫]) + (1/𝑟sin(𝜃))(∂/∂𝜃[sin(𝜃)𝛉]) + (1/𝑟sin(𝜃))(∂𝛗/∂𝜑)
    '''
    i = vec.dot(S.i)
    j = vec.dot(S.j)
    k = vec.dot(S.k)

    radial_term = 1/r**2 * diff(r**2 * i, r)
    theta_term = 1/(r*sin(theta)) * diff(sin(theta)*j)
    phi_term = 1/(r*sin(theta)) * diff(k, phi)

    return radial_term + theta_term + phi_term


def S_curl(vec):
    '''
    The curl in spherical coordinates.

    For a vector <𝐫, 𝛉, 𝛗>, the curl in spherical coordinates is given by:
    𝝯x𝑓 = (1/𝑟sin(𝜃))[(∂/∂𝜃(sin(𝜃)𝛗)) - (∂𝛉/∂𝜑)]𝐫 + (1/𝑟)[(1/sin(𝜃))(∂𝐫/∂𝜑) - (∂𝛗/∂𝑟)]𝛉 + (1/𝑟)[(∂/∂𝑟(𝑟𝛉)) - (∂𝐫/∂𝜃)]𝛗
    '''
    i = vec.dot(S.i)
    j = vec.dot(S.j)
    k = vec.dot(S.k)

    radial_term = 1/(r*sin(theta)) * (diff(k * sin(theta), theta) - diff(j, phi))
    theta_term = 1/r * (1/sin(theta) * diff(i, r) - diff(r* k, r))
    phi_term = 1/r * (diff(r*j, r) - diff(i, theta))

    return radial_term * S.i + theta_term * S.j + phi_term * S.k
    

def S_lapl(fun):
    '''
    The Laplacian in spherical coordinates.

    For a scalar function 𝑓, the Laplacian in spherical coordinates is given by:
    𝝯²𝑓 = (1/𝑟²)(∂/∂𝑟[𝑟²(∂𝑓/∂𝑟)]) + (1/𝑟²sin(𝜃))(∂/∂𝜃[sin(𝜃)(∂𝑓/∂𝜃)]) + (1/𝑟²sin²(𝜃))(∂²𝑓/∂𝜑²)
    '''
    dif = diff(fun, r)
    radial_term = (1/r**2) * diff(r**2 * dif, r)
    dif = diff(fun, theta)
    theta_term = 1/(r**2 * sin(theta)) * diff(sin(theta) * dif, theta)
    dif = diff(fun, phi)
    phi_term = 1/(r**2 * sin(theta)**2) * diff(dif, phi)

    return radial_term + theta_term + phi_term
