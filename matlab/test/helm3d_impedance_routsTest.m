%
% This file tests the Helmholtz impedance problem
%
%
run ../startup.m
S = geometries.sphere(1, 2, [0;0;0], 4, 1);

tic, [srcvals,~,~,~,~,wts] = extract_arrays(S); toc;
[~, npts] = size(srcvals);

zk = 1.1;
alpha = 1.0;

eps = 1e-7;

xyz_in = [0.3;0.5;0.1];
xyz_out = [1.3;-5.2;0.1];
src_info = [];
src_info.r = xyz_in;

zlams = complex(randn(npts,1));
rhs = helm3d.kern(zk, src_info, S, 'sprime') + ...
   1j*zk.*zlams.*helm3d.kern(zk, src_info, S, 's');

[densities] = helm3d.impedance.solver(S, zlams, rhs, eps, zk, alpha);

targ_info = [];
targ_info.r = xyz_out;

pot = helm3d.impedance.eval(S, densities, targ_info, eps, zk, alpha);
pot_ex = helm3d.kern(zk, src_info,targ_info,'s');
fprintf('Error in iterative solver=%d\n',abs(pot-pot_ex)/abs(pot_ex));

