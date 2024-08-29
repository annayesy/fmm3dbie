program surf_refine_test

  implicit double precision (a-h,o-z) 
  double precision, allocatable :: srcvals(:,:),srccoefs(:,:)
  double precision, allocatable :: wts(:)
  character *100 :: fname
  integer :: ipars(2)

  integer, allocatable :: norders(:),ixyzs(:),iptype(:)

  double precision :: xyz_out(3), xyz_in(3), evec(3)
  double complex, allocatable :: sigma(:),rhs(:)
  double precision, allocatable :: errs(:)
  double precision eps_gmres
  complex * 16 zpars(3)
  integer numit,niter

  integer ipatch_id, ilist(100)
  double precision uvs_targ(2)

  logical isout0,isout1

  double complex :: pot, potex, ztmp, ima, cd

  data ima/(0.0d0,1.0d0)/


  call prini(6,13)

  done = 1
  pi = atan(done)*4

  
  !
  ! select geometry type
  !   igeomtype = 1  =>  sphere
  !   igeomtype = 2  =>  stellarator
  !c 
  igeomtype = 1
  
  if(igeomtype.eq.1) then
    ipars(1) = 0
    npatches = 12*(4**ipars(1))
  end if
  
  if(igeomtype.eq.2) then
    ipars(1) = 10
    ipars(2) = ipars(1)*3
    npatches = 2*ipars(1)*ipars(2)
  end if
  

  call prinf('npatches = *', npatches, 1)

  npatmax = 10*npatches
  norder = 4
  npols = (norder+1)*(norder+2)/2

  call prinf('npols = *', npols, 1)
  
  npts = npatches*npols
  call prinf('npts = *', npts, 1)

  nptmax = 10*npts

  allocate(srcvals(12,nptmax),srccoefs(9,nptmax))
  ifplot = 1

  !
  ! construct the surface data
  !
  fname = 'surface3.vtk'
  call setup_geom(igeomtype, norder, npatches, ipars, &
      srcvals, srccoefs, ifplot, fname)

  
  allocate(norders(npatmax),ixyzs(npatmax+1),iptype(npatmax))

  do i=1,npatches
    norders(i) = norder
    ixyzs(i) = 1 +(i-1)*npols
    iptype(i) = 1
  enddo

  ixyzs(npatches+1) = 1+npols*npatches
  allocate(wts(nptmax))
  call get_qwts(npatches,norders,ixyzs,iptype,npts,srcvals,wts)
  print *, sum(wts(1:npts))-4*pi


  call surf_quadratic_msh_vtk_plot(npatches, norders, ixyzs, iptype, &
    npts, srccoefs, srcvals, 'sphere_msh.vtk', 'a')

  nlist = 2
  ilist(1) = 1
  ilist(2) = 5

  ier = 0
  call refine_patches_list(npatches, npatmax, norders, ixyzs, &
    iptype, npts, nptmax, srccoefs, srcvals, nlist, ilist, &
    npatches_out, npts_out, ier)

  print *, npatches_out, npts_out
  
  call surf_quadratic_msh_vtk_plot(npatches_out, norders, ixyzs, iptype, &
    npts_out, srccoefs, srcvals, 'sphere_msh_refine.vtk', 'a')


  nlist = 1
  ilist(1) = 13
  
  ier = 0
  npatches = npatches_out
  npts = npts_out
  call refine_patches_list(npatches, npatmax, norders, ixyzs, &
    iptype, npts, nptmax, srccoefs, srcvals, nlist, ilist, &
    npatches_out, npts_out, ier)

  
  call surf_quadratic_msh_vtk_plot(npatches_out, norders, ixyzs, iptype, &
    npts_out, srccoefs, srcvals, 'sphere_msh_refine2.vtk', 'a')

  call get_qwts(npatches_out,norders,ixyzs,iptype,npts_out,srcvals,wts)

  print *, sum(wts(1:npts_out))-4*pi


    
end program surf_refine_test 

!----------------------------------------------------------------
!
! this is the end of the driver, geometry routines are below
!
!----------------------------------------------------------------



subroutine setup_geom(igeomtype, norder, npatches, ipars, &
    srcvals, srccoefs, ifplot, fname)
  implicit double precision (a-h,o-z)
  integer :: igeomtype,norder,npatches,ipars(*),ifplot
  character (len=*) :: fname
  double precision :: srcvals(12,*), srccoefs(9,*)

  double precision, allocatable :: uvs(:,:),umatr(:,:)
  double precision, allocatable :: vmatr(:,:),wts(:)

  double precision :: xyz(10), dxyzduv(3,10)
  double precision :: xyz1(10), dxyzduv1(3,10)
  double precision :: xyz2(10), dxyzduv2(3,10)
  double precision, target :: scales(10)
  double precision, pointer :: ptr1,ptr2,ptr3,ptr4
  integer, pointer :: iptr1, iptr2, iptr3, iptr4
  double precision, target :: p1(10),p2(10),p3(10),p4(10)
  double precision, allocatable, target :: triaskel(:,:,:)
  double precision, allocatable, target :: deltas(:,:)
  integer, allocatable :: isides(:)
  integer, target :: nmax,mmax

  procedure (), pointer :: xtri_geometry


  external xtri_stell_eval, xtri_sphere_eval
  external xtri_taco_eval, xtri_potato_eval
  external xtri_ellipsoid_eval

  npols = (norder+1)*(norder+2)/2
  allocate(uvs(2,npols),umatr(npols,npols),vmatr(npols,npols))
  allocate(wts(npols))

  call vioreanu_simplex_quad(norder,npols,uvs,umatr,vmatr,wts)

  if (igeomtype .eq. 1) then

    itype = 2
    allocate(triaskel(3,3,npatches))
    allocate(isides(npatches))
    npmax = npatches
    ntri = 0

    call xtri_platonic(itype, ipars(1), npmax, ntri,  &
        triaskel, isides)

    scales(1) = 0.5d0
    scales(2) = 1/(4*sqrt(5.0d0))
    scales(3) = 1/(4*sqrt(10.0d0))
    
    xtri_geometry => xtri_sphere_eval
    ptr1 => triaskel(1,1,1)
    ptr2 => scales(1)
    ptr3 => p3(1)
    ptr4 => p4(1)
    
    if(ifplot.eq.1) then
      call xtri_vtk_surf(fname,npatches,xtri_geometry, ptr1,ptr2,&
          ptr3,ptr4, norder,'Triangulated surface of the ellipsoid')
    endif

    call getgeominfo(npatches,xtri_geometry,ptr1,ptr2,ptr3,ptr4, &
        npols,uvs,umatr,srcvals,srccoefs)
  endif

  !
  ! set up a rectangular parameter domain and evaluate the stellarator
  !
  if(igeomtype.eq.2) then
    done = 1
    pi = atan(done)*4
    umin = 0
    umax = 2*pi
    vmin = 2*pi
    vmax = 0
    allocate(triaskel(3,3,npatches))
    nover = 0
    call xtri_rectmesh_ani(umin,umax,vmin,vmax,ipars(1),ipars(2), &
        nover,npatches,npatches,triaskel)

    mmax = 2
    nmax = 1
    xtri_geometry => xtri_stell_eval

    allocate(deltas(-1:mmax,-1:nmax))
    deltas(-1,-1) = 0.17d0
    deltas(0,-1) = 0
    deltas(1,-1) = 0
    deltas(2,-1) = 0

    deltas(-1,0) = 0.11d0
    deltas(0,0) = 1
    deltas(1,0) = 4.5d0
    deltas(2,0) = -0.25d0

    deltas(-1,1) = 0
    deltas(0,1) = 0.07d0
    deltas(1,1) = 0
    deltas(2,1) = -0.45d0

    ptr1 => triaskel(1,1,1)
    ptr2 => deltas(-1,-1)
    iptr3 => mmax
    iptr4 => nmax

    if(ifplot .eq. 1) then
      call prinf('plotting, npatches = *', npatches, 1)
      call prinf('plotting, norder = *', norder, 1)
      call xtri_vtk_surf(fname,npatches,xtri_geometry, ptr1,ptr2, &
          iptr3,iptr4, norder, &
          'Triangulated surface of the stellarator')
    endif

    call getgeominfo(npatches,xtri_geometry,ptr1,ptr2,iptr3,iptr4,&
        npols,uvs,umatr,srcvals,srccoefs)
  endif


  !
  ! setup Tom Hagstrom's potato taco
  !
  if(igeomtype .eq. 3) then

    itype = 2
    allocate(triaskel(3,3,npatches))
    allocate(isides(npatches))
    npmax = npatches
    ntri = 0

    call xtri_platonic(itype, ipars(1), npmax, ntri,  &
        triaskel, isides)

    xtri_geometry => xtri_potato_eval
    ptr1 => triaskel(1,1,1)
    ptr2 => p2(1)
    ptr3 => p3(1)
    ptr4 => p4(1)
    
    if(ifplot.eq.1) then
      call xtri_vtk_surf(fname,npatches,xtri_geometry, ptr1,ptr2,&
          ptr3,ptr4, norder,'Triangulated surface of the potato')
    endif

    
    call getgeominfo(npatches,xtri_geometry,ptr1,ptr2,ptr3,ptr4, &
        npols,uvs,umatr,srcvals,srccoefs)

!!!!!!!!!!!!!!!!!!!!!!!!1



    done = 1
    pi = atan(done)*4
    umin = 0.1d0
    umax = 0.9d0
    !ipars(1) = ipars(1)/2
    vmin = -pi
    !vmin = 0
    vmax = pi

    allocate(triaskel(3,3,npatches))
    nover = 0
    call xtri_rectmesh_ani(umin, umax, vmin, vmax, ipars(1), ipars(2), &
        nover, npatches, npatches, triaskel)

    xtri_geometry => xtri_taco_eval
    !xtri_geometry => xtri_potato_eval
    ptr1 => triaskel(1,1,1)
    ptr2 => p2(1)
    ptr3 => p3(1)
    ptr4 => p4(1)

    itri = 1
    u0 = 0.2d0
    v0 = .15d0
    call xtri_potato_eval(itri, u0, v0, xyz, dxyzduv, triaskel, &
        ptr2, ptr3, ptr4)

    call prinf('for itri = *', itri, 1)
    call prin2('at u0 = *', u0, 1)
    call prin2('and v0 = *', v0, 1)
    call prin2('xyz = *', xyz, 3)
    call prin2('dxyzduv = *', dxyzduv, 6)

    print *
    print *
    h = 0.001d0
    u = u0 + h
    v = v0
    call xtri_potato_eval(itri, u, v, xyz2, dxyzduv2, triaskel, &
        ptr2, ptr3, ptr4)

    u = u0 - h
    v = v0
    call xtri_potato_eval(itri, u, v, xyz1, dxyzduv1, triaskel, &
        ptr2, ptr3, ptr4)

    dxyzduv1(1,1) = (xyz2(1) - xyz1(1))/2/h
    dxyzduv1(2,1) = (xyz2(2) - xyz1(2))/2/h
    dxyzduv1(3,1) = (xyz2(3) - xyz1(3))/2/h
    call prin2('from finite diff, du ders = *', dxyzduv1, 3)

    dxyzduv1(1,1) = dxyzduv(1,1) - dxyzduv1(1,1)
    dxyzduv1(2,1) = dxyzduv(2,1) - dxyzduv1(2,1)
    dxyzduv1(3,1) = dxyzduv(3,1) - dxyzduv1(3,1)
    call prin2('and du errors = *', dxyzduv1, 3)
    
    
    u = u0
    v = v0+h
    call xtri_potato_eval(itri, u, v, xyz2, dxyzduv2, triaskel, &
        ptr2, ptr3, ptr4)

    u = u0
    v = v0 - h
    call xtri_potato_eval(itri, u, v, xyz1, dxyzduv1, triaskel, &
        ptr2, ptr3, ptr4)

    dxyzduv1(1,2) = (xyz2(1) - xyz1(1))/2/h
    dxyzduv1(2,2) = (xyz2(2) - xyz1(2))/2/h
    dxyzduv1(3,2) = (xyz2(3) - xyz1(3))/2/h
    call prin2('from finite diff, dv ders = *', dxyzduv1(1,2), 3)
    
    dxyzduv1(1,2) = dxyzduv(1,2) - dxyzduv1(1,2)
    dxyzduv1(2,2) = dxyzduv(2,2) - dxyzduv1(2,2)
    dxyzduv1(3,2) = dxyzduv(3,2) - dxyzduv1(3,2)
    call prin2('and dv errors = *', dxyzduv1(1,2), 3)

    
    if(ifplot .eq. 1) then
      call prinf('plotting, npatches = *', npatches, 1)
      call prinf('plotting, norder = *', norder, 1)
      call xtri_vtk_surf(fname, npatches, xtri_geometry, ptr1, ptr2, &
          iptr3, iptr4, norder, &
          'Triangulated surface of the Toms taco')
    endif

    
    call getgeominfo(npatches, xtri_geometry, ptr1, ptr2, &
        iptr3, iptr4, npols, uvs, umatr, srcvals, srccoefs)
  endif
  
  
  return  
end subroutine setup_geom





subroutine xtri_ellipsoid_eval(itri, u, v, xyz, dxyzduv, triainfo, &
    scales, p3, p4)
  implicit real *8 (a-h,o-z)
  double precision :: xyz(3), dxyzduv(3,2), triainfo(3,3,*)
  double precision :: scales(3)

  !
  ! project the triangle itri in triainfo onto the sphere
  !
  !    Input:
  ! itri - triangle number to map
  ! u,v - local uv coordinates on triangle itri
  ! triainfo - flat skeleton triangle info
  ! p2,p3,p4 - dummy parameters
  !
  !    Output:
  ! xyz - point on the sphere
  ! dxyzduv - first derivative information
  !
  !

  x0=triainfo(1,1,itri)
  y0=triainfo(2,1,itri)
  z0=triainfo(3,1,itri)

  x1=triainfo(1,2,itri)
  y1=triainfo(2,2,itri)
  z1=triainfo(3,2,itri)

  x2=triainfo(1,3,itri)
  y2=triainfo(2,3,itri)
  z2=triainfo(3,3,itri)

  !
  ! ... process the geometry, return the point location on the sphere
  ! and the derivatives with respect to u and v
  !
  x=x0+u*(x1-x0)+v*(x2-x0)
  y=y0+u*(y1-y0)+v*(y2-y0)
  z=z0+u*(z1-z0)+v*(z2-z0)

  dxdu = x1-x0
  dydu = y1-y0
  dzdu = z1-z0

  dxdv = x2-x0
  dydv = y2-y0
  dzdv = z2-z0

  !
  ! second derivatives are zero...
  !

  !
  ! project onto the sphere
  !
  phi = atan2(y, x)
  theta = atan2(sqrt(x**2 + y**2), z)
  
  e = sin(theta)**2 * cos(phi)**2 / scales(1)**2
  f = sin(theta)**2 * sin(phi)**2 / scales(2)**2
  g = cos(theta)**2  / scales(3)**2

  r = sqrt(1/(e + f + g))
  xyz(1) = r*sin(theta)*cos(phi)
  xyz(2) = r*sin(theta)*sin(phi)
  xyz(3) = r*cos(theta)

  !xyz(1) = phi
  !xyz(2) = theta
  !xyz(3) = r
  

  ! and now the derivatives
  dphidu = 1/(x**2 + y**2) * (x*dydu - y*dxdu)
  dphidv = 1/(x**2 + y**2) * (x*dydv - y*dxdv)

  p = sqrt(x**2+y**2)
  dthetadu = (z/p*(x*dxdu + y*dydu) - dzdu*p)/(x**2 + y**2 + z**2)
  dthetadv = (z/p*(x*dxdv + y*dydv) - dzdv*p)/(x**2 + y**2 + z**2)

  dedu = 1/scales(1)**2 &
      * (2*sin(theta)*cos(theta)*dthetadu*cos(phi)**2 &
      - 2*sin(theta)**2 * cos(phi)*sin(phi)*dphidu)
  dfdu = 1/scales(2)**2 &
      * (2*sin(theta)*cos(theta)*dthetadu*sin(phi)**2 &
      + 2*sin(theta)**2 * sin(phi)*cos(phi)*dphidu)
  dgdu = -1/scales(3)**2 * 2*cos(theta)*sin(theta)*dthetadu
  
  dedv = 1/scales(1)**2 &
      * (2*sin(theta)*cos(theta)*dthetadv*cos(phi)**2 &
      - 2*sin(theta)**2 * cos(phi)*sin(phi)*dphidv)
  dfdv = 1/scales(2)**2 &
      * (2*sin(theta)*cos(theta)*dthetadv*sin(phi)**2 &
      + 2*sin(theta)**2 * sin(phi)*cos(phi)*dphidv)
  dgdv = -1/scales(3)**2 * 2*cos(theta)*sin(theta)*dthetadv
  
  drdu = -(dedu + dfdu + dgdu)/2/r/(e+f+g)**2
  drdv = -(dedv + dfdv + dgdv)/2/r/(e+f+g)**2


  ! now for the final derivatives
  dxyzduv(1,1) = drdu*sin(theta)*cos(phi) &
      + r*cos(theta)*dthetadu*cos(phi) &
      - r*sin(theta)*sin(phi)*dphidu

  dxyzduv(2,1) = drdu*sin(theta)*sin(phi) &
      + r*cos(theta)*dthetadu*sin(phi) &
      + r*sin(theta)*cos(phi)*dphidu

  dxyzduv(3,1) = drdu*cos(theta) - r*sin(theta)*dthetadu
  
  dxyzduv(1,2) = drdv*sin(theta)*cos(phi) &
      + r*cos(theta)*dthetadv*cos(phi) &
      - r*sin(theta)*sin(phi)*dphidv

  dxyzduv(2,2) = drdv*sin(theta)*sin(phi) &
      + r*cos(theta)*dthetadv*sin(phi) &
      + r*sin(theta)*cos(phi)*dphidv

  dxyzduv(3,2) = drdv*cos(theta) - r*sin(theta)*dthetadv
  
  !dxyzduv(1,1) = dphidu
  !dxyzduv(2,1) = dthetadu
  !dxyzduv(3,1) = drdu

  !dxyzduv(1,2) = dphidv
  !dxyzduv(2,2) = dthetadv
  !dxyzduv(3,2) = drdv

  
  
  ! r=sqrt(x**2+y**2+z**2)
  ! xyz(1)=x/r
  ! xyz(2)=y/r
  ! xyz(3)=z/r

  ! a = x0*(x1-x0) + y0*(y1-y0) + z0*(z1-z0)
  ! b = (x1-x0)*(x2-x0) + (y1-y0)*(y2-y0) + (z1-z0)*(z2-z0)
  ! c = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0)

  ! drdu = (a + v*b + u*c)/r
  ! drdu2 = (r*c - r*drdu*drdu)/r/r

  ! e = x0*(x2-x0) + y0*(y2-y0) + z0*(z2-z0)
  ! f = b
  ! g = (x2-x0)*(x2-x0) + (y2-y0)*(y2-y0) + (z2-z0)*(z2-z0)

  ! drdv = (e + u*f + v*g)/r
  ! drdv2 = (r*g - r*drdv*drdv)/r/r

  ! drduv = (r*b - r*drdu*drdv)/r/r

  ! ! du
  ! dxyzduv(1,1) = (r*dxdu-x*drdu)/r/r
  ! dxyzduv(2,1) = (r*dydu-y*drdu)/r/r
  ! dxyzduv(3,1) = (r*dzdu-z*drdu)/r/r

  ! ! dv
  ! dxyzduv(1,2) = (r*dxdv-x*drdv)/r/r
  ! dxyzduv(2,2) = (r*dydv-y*drdv)/r/r
  ! dxyzduv(3,2) = (r*dzdv-z*drdv)/r/r

  return
end subroutine xtri_ellipsoid_eval








subroutine xtri_taco_eval(itri, u, v, xyz, dxyzduv, triainfo, &
    p2, p3, p4)
  implicit real *8 (a-h,o-z)
  real *8 :: xyz(3), dxyzduv(3,2), triainfo(3,3,*)

  !
  ! project the triangle itri in triainfo onto Tom Hagstrom's taco
  !
  !    Input:
  ! itri - triangle number to map
  ! u,v - local uv coordinates on triangle itri
  ! triainfo - flat skeleton triangle info
  ! p2,p3,p4 - dummy parameters
  !
  !    Output:
  ! xyz - point on the taco
  ! dxyzduv - first derivative information
  !
  !

  x0=triainfo(1,1,itri)
  y0=triainfo(2,1,itri)
  z0=triainfo(3,1,itri)

  x1=triainfo(1,2,itri)
  y1=triainfo(2,2,itri)
  z1=triainfo(3,2,itri)

  x2=triainfo(1,3,itri)
  y2=triainfo(2,3,itri)
  z2=triainfo(3,3,itri)

  !
  ! ... process the geometry, return the point location on the sphere
  ! and the derivatives with respect to u and v
  !
  s = x0+u*(x1-x0)+v*(x2-x0)
  t = y0+u*(y1-y0)+v*(y2-y0)

  r = 2*s*(1-s)/5/(cos(t)**6 + sin(t)**6)**(1/6.0d0)

  sqr3 = sqrt(3.0d0)
  done = 1
  pi = 4*atan(done)
  x = (2*s-1)/4 + sqr3*r*sin(t)/2
  y = (sqr3/4 - r*cos(t))*cos(pi*s/2) + r*sin(t)*sin(pi*s/2)/2
  z = (sqr3/4 - r*cos(t))*sin(pi*s/2) - r*sin(t)*cos(pi*s/2)/2

  xyz(1) = x
  xyz(2) = y
  xyz(3) = z
  
  !
  ! and now the derivatives
  !
  dsdu = x1-x0
  dtdu = y1-y0

  dsdv = x2-x0
  dtdv = y2-y0

  drds = 2*(1 - 2*s)/5/(cos(t)**6 + sin(t)**6)**(1/6.0d0)
  drdt = .4d0 * s*(1-s) * (cos(t)**6 + sin(t)**6)**(-7/6.0d0) &
      *( cos(t)**5 * sin(t) - sin(t)**5 * cos(t) )
  
  drdu = drds*dsdu + drdt*dtdu
  drdv = drds*dsdv + drdt*dtdv

  dxdu = dsdu/2 + sqr3/2*( drdu*sin(t) + r*cos(t)*dtdu )
  dydu = (-drdu*cos(t) + r*sin(t)*dtdu)*cos(pi*s/2) &
      - (sqr3/4 - r*cos(t))*pi/2*sin(pi*s/2)*dsdu &
      + .5d0 * drdu*sin(t)*sin(pi*s/2) &
      + .5d0 * r*cos(t)*dtdu*sin(pi*s/2) &
      + .5d0 * r*sin(t)*cos(pi*s/2)*pi/2*dsdu
  dzdu = (-drdu*cos(t) + r*sin(t)*dtdu)*sin(pi*s/2) &
      + (sqr3/4 - r*cos(t))*pi/2*cos(pi*s/2)*dsdu &
      - .5d0 * drdu*sin(t)*cos(pi*s/2) &
      - .5d0 * r*cos(t)*dtdu*cos(pi*s/2) &
      + .5d0 * r*sin(t)*sin(pi*s/2)*pi/2*dsdu
      
  dxdv = dsdv/2 + sqr3/2*( drdv*sin(t) + r*cos(t)*dtdv )
  dydv = (-drdv*cos(t) + r*sin(t)*dtdv)*cos(pi*s/2) &
      - (sqr3/4 - r*cos(t))*pi/2*sin(pi*s/2)*dsdv &
      + .5d0 * drdv*sin(t)*sin(pi*s/2) &
      + .5d0 * r*cos(t)*dtdv*sin(pi*s/2) &
      + .5d0 * r*sin(t)*cos(pi*s/2)*pi/2*dsdv
  dzdv = (-drdv*cos(t) + r*sin(t)*dtdv)*sin(pi*s/2) &
      + (sqr3/4 - r*cos(t))*pi/2*cos(pi*s/2)*dsdv &
      - .5d0 * drdv*sin(t)*cos(pi*s/2) &
      - .5d0 * r*cos(t)*dtdv*cos(pi*s/2) &
      + .5d0 * r*sin(t)*sin(pi*s/2)*pi/2*dsdv
      
  ! du
  dxyzduv(1,1) = dxdu
  dxyzduv(2,1) = dydu
  dxyzduv(3,1) = dzdu

  ! dv
  dxyzduv(1,2) = dxdv
  dxyzduv(2,2) = dydv
  dxyzduv(3,2) = dzdv

  return



  !
  ! second derivatives are zero...
  !

  !
  ! project onto the taco
  !

  ! a = x0*(x1-x0) + y0*(y1-y0) + z0*(z1-z0)
  ! b = (x1-x0)*(x2-x0) + (y1-y0)*(y2-y0) + (z1-z0)*(z2-z0)
  ! c = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0)

  ! drdu = (a + v*b + u*c)/r
  ! drdu2 = (r*c - r*drdu*drdu)/r/r

  ! e = x0*(x2-x0) + y0*(y2-y0) + z0*(z2-z0)
  ! f = b
  ! g = (x2-x0)*(x2-x0) + (y2-y0)*(y2-y0) + (z2-z0)*(z2-z0)

  ! drdv = (e + u*f + v*g)/r
  ! drdv2 = (r*g - r*drdv*drdv)/r/r

  ! drduv = (r*b - r*drdu*drdv)/r/r

  ! ! du
  ! dxyzduv(1,1) = (r*dxdu-x*drdu)/r/r
  ! dxyzduv(2,1) = (r*dydu-y*drdu)/r/r
  ! dxyzduv(3,1) = (r*dzdu-z*drdu)/r/r

  ! ! dv
  ! dxyzduv(1,2) = (r*dxdv-x*drdv)/r/r
  ! dxyzduv(2,2) = (r*dydv-y*drdv)/r/r
  ! dxyzduv(3,2) = (r*dzdv-z*drdv)/r/r

  return
end subroutine xtri_taco_eval





subroutine xtri_potato_eval(itri, u, v, xyz, dxyzduv, triainfo, &
    p2, p3, p4)
  implicit real *8 (a-h,o-z)
  real *8 :: xyz(3), dxyzduv(3,2), triainfo(3,3,*)

  !
  ! project the triangle itri in triainfo onto Tom Hagstrom's taco
  !
  !    Input:
  ! itri - triangle number to map
  ! u,v - local uv coordinates on triangle itri
  ! triainfo - flat skeleton triangle info
  ! p2,p3,p4 - dummy parameters
  !
  !    Output:
  ! xyz - point on the taco
  ! dxyzduv - first derivative information
  !
  !

  x0=triainfo(1,1,itri)
  y0=triainfo(2,1,itri)
  z0=triainfo(3,1,itri)

  x1=triainfo(1,2,itri)
  y1=triainfo(2,2,itri)
  z1=triainfo(3,2,itri)

  x2=triainfo(1,3,itri)
  y2=triainfo(2,3,itri)
  z2=triainfo(3,3,itri)

  !
  ! ... process the geometry, return the point location on the sphere
  ! and the derivatives with respect to u and v
  !

  ! get the points on the skeleton mesh
  x = x0+u*(x1-x0)+v*(x2-x0)
  y = y0+u*(y1-y0)+v*(y2-y0)
  z = z0+u*(z1-z0)+v*(z2-z0)

  ! and their partial derivatives
  dxdu = x1-x0
  dydu = y1-y0
  dzdu = z1-z0

  dxdv = x2-x0
  dydv = y2-y0
  dzdv = z2-z0

  ! now the spherical angles
  phi = atan2(y, x)
  theta = atan2(sqrt(x**2 + y**2), z)

  ! and their partial derivatives
  dphidu = 1/(x**2 + y**2) * (x*dydu - y*dxdu)
  dphidv = 1/(x**2 + y**2) * (x*dydv - y*dxdv)

  p = sqrt(x**2+y**2)
  dthetadu = (z/p*(x*dxdu + y*dydu) - dzdu*p)/(x**2 + y**2 + z**2)
  dthetadv = (z/p*(x*dxdv + y*dydv) - dzdv*p)/(x**2 + y**2 + z**2)

  ! now compute the points on the surface
  done = 1
  pi = 4*atan(done)
  s = theta/pi
  dsdu = (1/pi)*dthetadu
  dsdv = (1/pi)*dthetadv

  d = (cos(phi)**6 + sin(phi)**6)**(1/6.0d0)
  r = .4d0 * sqrt(s*(1-s))/d

  
  
  sqr3 = sqrt(3.0d0)
  xxx = (2*s-1)/4 + sqr3*r*sin(t)/2
  yyy = (sqr3/4 - r*cos(phi))*cos(pi*s/2) + r*sin(phi)*sin(pi*s/2)/2
  zzz = (sqr3/4 - r*cos(phi))*sin(pi*s/2) - r*sin(phi)*cos(pi*s/2)/2

  xyz(1) = xxx
  xyz(2) = yyy
  xyz(3) = zzz

  ! now compute the partial derivatives of the points on the surface
  

  
  return
  !
  ! and now the derivatives
  !
  dsdu = x1-x0
  dtdu = y1-y0

  dsdv = x2-x0
  dtdv = y2-y0

  drds = .2d0 * (1 - 2*s)/sqrt(s-s*s)/(cos(t)**6 + sin(t)**6)**(1/6.0d0)
  drdt = .4d0 * sqrt(s*(1-s)) * (cos(t)**6 + sin(t)**6)**(-7/6.0d0) &
      *( cos(t)**5 * sin(t) - sin(t)**5 * cos(t) )
  
  drdu = drds*dsdu + drdt*dtdu
  drdv = drds*dsdv + drdt*dtdv

  dxdu = dsdu/2 + sqr3/2*( drdu*sin(t) + r*cos(t)*dtdu )
  dydu = (-drdu*cos(t) + r*sin(t)*dtdu)*cos(pi*s/2) &
      - (sqr3/4 - r*cos(t))*pi/2*sin(pi*s/2)*dsdu &
      + .5d0 * drdu*sin(t)*sin(pi*s/2) &
      + .5d0 * r*cos(t)*dtdu*sin(pi*s/2) &
      + .5d0 * r*sin(t)*cos(pi*s/2)*pi/2*dsdu
  dzdu = (-drdu*cos(t) + r*sin(t)*dtdu)*sin(pi*s/2) &
      + (sqr3/4 - r*cos(t))*pi/2*cos(pi*s/2)*dsdu &
      - .5d0 * drdu*sin(t)*cos(pi*s/2) &
      - .5d0 * r*cos(t)*dtdu*cos(pi*s/2) &
      + .5d0 * r*sin(t)*sin(pi*s/2)*pi/2*dsdu
      
  dxdv = dsdv/2 + sqr3/2*( drdv*sin(t) + r*cos(t)*dtdv )
  dydv = (-drdv*cos(t) + r*sin(t)*dtdv)*cos(pi*s/2) &
      - (sqr3/4 - r*cos(t))*pi/2*sin(pi*s/2)*dsdv &
      + .5d0 * drdv*sin(t)*sin(pi*s/2) &
      + .5d0 * r*cos(t)*dtdv*sin(pi*s/2) &
      + .5d0 * r*sin(t)*cos(pi*s/2)*pi/2*dsdv
  dzdv = (-drdv*cos(t) + r*sin(t)*dtdv)*sin(pi*s/2) &
      + (sqr3/4 - r*cos(t))*pi/2*cos(pi*s/2)*dsdv &
      - .5d0 * drdv*sin(t)*cos(pi*s/2) &
      - .5d0 * r*cos(t)*dtdv*cos(pi*s/2) &
      + .5d0 * r*sin(t)*sin(pi*s/2)*pi/2*dsdv
      
  ! du
  dxyzduv(1,1) = dxdu
  dxyzduv(2,1) = dydu
  dxyzduv(3,1) = dzdu

  ! dv
  dxyzduv(1,2) = dxdv
  dxyzduv(2,2) = dydv
  dxyzduv(3,2) = dzdv

  return



  !
  ! second derivatives are zero...
  !

  !
  ! project onto the taco
  !

  ! a = x0*(x1-x0) + y0*(y1-y0) + z0*(z1-z0)
  ! b = (x1-x0)*(x2-x0) + (y1-y0)*(y2-y0) + (z1-z0)*(z2-z0)
  ! c = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0)

  ! drdu = (a + v*b + u*c)/r
  ! drdu2 = (r*c - r*drdu*drdu)/r/r

  ! e = x0*(x2-x0) + y0*(y2-y0) + z0*(z2-z0)
  ! f = b
  ! g = (x2-x0)*(x2-x0) + (y2-y0)*(y2-y0) + (z2-z0)*(z2-z0)

  ! drdv = (e + u*f + v*g)/r
  ! drdv2 = (r*g - r*drdv*drdv)/r/r

  ! drduv = (r*b - r*drdu*drdv)/r/r

  ! ! du
  ! dxyzduv(1,1) = (r*dxdu-x*drdu)/r/r
  ! dxyzduv(2,1) = (r*dydu-y*drdu)/r/r
  ! dxyzduv(3,1) = (r*dzdu-z*drdu)/r/r

  ! ! dv
  ! dxyzduv(1,2) = (r*dxdv-x*drdv)/r/r
  ! dxyzduv(2,2) = (r*dydv-y*drdv)/r/r
  ! dxyzduv(3,2) = (r*dzdv-z*drdv)/r/r

  return
end subroutine xtri_potato_eval





subroutine test_exterior_pt(npatches,norder,npts,srcvals, &
    srccoefs,wts,xyzout,isout)
  implicit none
  integer npatches,norder,npts
  ! c
  ! c  this subroutine tests whether the pt xyzin, is
  ! c  in the exterior of a surface, and also estimates the error
  ! c  in representing e^{ir/2}/r and \grad e^{ir/2}/r \cdot n
  ! c  centered at the interior point. Whether a point 
  ! c  is in the interior or not is tested using Gauss' 
  ! c  identity for the flux due to a point charge
  ! c
  ! c
  ! c  input:
  ! c    npatches - integer
  ! c       number of patches
  ! c    norder - integer
  ! c       order of discretization
  ! c    npts - integer
  ! c       total number of discretization points on the surface
  ! c    srccoefs - double precision (9,npts)
  ! c       koornwinder expansion coefficients of geometry info
  ! c    xyzout -  double precision (3)
  ! c       point to be tested
  ! c
  ! c  output: 
  ! c    isout - boolean
  ! c      whether the target is in the interior or not
  ! c
  integer :: npols
  double precision srccoefs(9,npts),srcvals(12,npts),xyzout(3),wts(npts)
  double precision tmp(3)
  double precision dpars,done,pi
  double precision, allocatable :: rsurf(:),err_p(:,:) 
  integer ipars,norderhead,nd
  double complex, allocatable :: sigma_coefs(:,:), sigma_vals(:,:)
  double complex zk,val

  integer ipatch,j,i
  double precision ra,ds
  logical isout

  done = 1
  pi = atan(done)*4

  npols = (norder+1)*(norder+2)/2
  zk = 0
  ra = 0


  do ipatch=1,npatches
    do j=1,npols
      i = (ipatch-1)*npols + j
      call h3d_sprime(xyzout,12,srcvals(1,i),0,dpars,1,zk,0,ipars,&
          val)

      call cross_prod3d(srcvals(4,i),srcvals(7,i),tmp)
      ds = sqrt(tmp(1)**2 + tmp(2)**2 + tmp(3)**2)
      ra = ra + real(val)*wts(i)
    enddo
  enddo

  if(abs(ra+4*pi).le.1.0d-1) isout = .false.
  if(abs(ra).le.1.0d-1) isout = .true.

  return
end subroutine test_exterior_pt

   




