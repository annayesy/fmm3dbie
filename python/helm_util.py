import numpy as np


def helm_dirichlet_kernel(xx,yy,zpars,nu):
	assert zpars.shape[0] == 3

	assert xx.shape[-1] == 3
	assert yy.shape[-1] == 3
	assert nu.shape[-1] == 3

	N = xx.shape[0]
	assert yy.shape[0] == N

	zk,alpha,beta = zpars

	dx0 = xx[:,0].reshape(N,1) - yy[:,0].reshape(1,N)
	dx1 = xx[:,1].reshape(N,1) - yy[:,1].reshape(1,N)
	dx2 = xx[:,2].reshape(N,1) - yy[:,2].reshape(1,N)

	dr  = np.sqrt(dx0 **2 + dx1**2 + dx2 **2)

	tmp_zeros = np.where(dr == 0)
	dr[tmp_zeros] = 1

	zexp  = np.exp(1j * zk * dr)

	rdotn = dx0 * nu[:,0] + dx1 * nu[:,1] + dx2 * nu[:,2]

	const = 1/(4*np.pi)

	single_lp = const * (zexp / dr)
	double_lp = const * (rdotn / (dr**3)) * zexp * (1.0 - 1j*zk * dr)
	K = alpha * single_lp + double_lp * beta

	K[tmp_zeros] = 0
	return K