import pytest
from ...z.input import InputElement, input_element_by_id


doc_test_cases = {
    0: [
        # 1 "step" where each step through the beamline element consists of a
        #   half-step +a space-charge kick + a half-step. Each half-step
        #   involves computing a map for that half-element, computed by
        #   numerical integration with 1 "map step", pipe radius is 1 m.
        "0.0620822983935 1 1 0 1.0 /",
    ],
    1: [
        # gradient=16.4423850936 Tesla/m, input gradient file ID (if >0, read
        # in fringe field profile; if<-10 use linear transfer map of an
        # undulator, if >-10 but <0, use k-value (i.e. Gradient/Brho) linear
        # transfer map; if=0 linear transfer map using gradient), radius = 1.0,
        # x misalignment error=0.0m, y misalignment error=0.0m, rotation error
        # x, y, z = 0.0, 0.0,
        # 0.0 rad.
        "0.05 3 1 1 16.4423850936 0 1.0 0. 0. 0. 0. 0. /",
    ],
    2: [
        # length=0.3m, 4 "steps", " 20 "map steps",kx0^2=9.8696, ky0^2=9.8696,
        # kz0^2=9.8696, radius=0.014m. Note, it does not work for Lorentz
        # integrator option.)
        "0.30 4 20 2 9.8696 9.8696 9.8696 0.014 /",
    ],
    # Solenoid
    3: [
        # length=0.3m, 4 "steps," 20 "map steps", Bz0=5.67 Tesla, input field
        # file ID 0., radius=0.014m, x misalignment error=0.0m, y misalignment
        # error=0.0m, rotation error x, y, z=0.0, 0., 0. rad. Note: For the
        # Lorentz integrator, the length includes two linear fringe regions and
        # a flat top region. Here, the length of the fringe region is defined
        # by the 2*radius. The total length = effective length + 2*radius.)
        "0.30 4 20 3 5.67 0. 0.014 0. 0. 0. 0. 0. /",
    ],
    # Dipole
    4: [
        # length=1.48524m, 10 "steps", 20 "map steps", bending angle 0.1rad,
        # k1=0.0, input switch 150., if > 200, include 1D CSR, half gap =
        # 0.014m, 0.0 = entr. pole face angle(rad), 0.0 = exit pole face angle,
        # 0.0 = curvature of entr. face, 0.0 = curv. of exit face, 0.0 =
        #   integrated fringe field, 0.0 = x misalignment error, 0.0 = y
        #   misalignment error, 0.0 = x rot. error, 0.0 = y rot. error, 0.0 = z
        #   rot error.
        "1.48524 10 20 4 0.1 0.0 150. 0.014 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 /",
    ],
    101: [
        # length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF
        # frequency=700.0e6, driven phase=30.0 degree, input field ID=1.0 (if
        # ID<0, uses simple sinusoidal model, only works for the map
        # integrator, phase is design phase with 0 for maximum energy gain),
        # radius=0.014m, quad 1 length=0.01m, quad 1 gradient=1.0T/m, quad 2
        # length=0.01m, quad 2 gradient, x misalignment error=0.0m, y
        # misalignment error=0.0m, rotation error x, y, z=0.0, 0., 0. rad for
        # quad, x displacement, y displacement, rotation error x, y, z for RF
        # field.
        "1.48524 10 20 101 1.0 700.0e6 30. 1.0 0.014 0.01 1.0 0.01 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 /",
    ],
    102: [
        # length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF
        # frequency=700.0e6, driven phase=30.0 degree, input field ID=1.0 (if
        # ID<0, use simple sinusoidal model, only works for the map integrator,
        # phase is design phase with 0 for maximum energy gain), radius=0.014m,
        # x misalignment error=0.0m, y misalignment error=0.0m, rotation error
        # x, y, z=0.0, 0., 0. Rad.
        "1.48524 10 20 102 1.0 700.0e6 30. 1.0 0.014 0. 0. 0. 0. 0. /",
    ],
    103: [
        # length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF
        # frequency=700.0e6, driven phase=30.0 degree, input field ID=1.0 (if
        # ID<0, use simple sinusoidal model, only works for the map integrator,
        # phase is design phase with 0 for maximum energy gain), radius=0.014m,
        # x misalignment error=0.0m, y misalignment error=0.0m, rotation error
        # x, y, z=0.0, 0., 0. Rad. e.g. 1.037743e+000 2 1 103 0.1090744783E+08
        # 1.3e9 -0.1743982832E+02 -1.01 1.0 / ("-1.01" for simple sinusoidal RF
        # cavity model.)
        "1.48524 10 20 103 1.0 700.0e6 30. 1.0 0.014 0. 0. 0. 0. 0. /",
        "1.037743e+000 2 1 103 0.1090744783E+08 1.3e9 -0.1743982832E+02 -1.01 1.0 /",
    ],
    104: [
        # 1 "map steps", field scaling=34000000.0, RF frequency=650.0e6, driven
        #   phase=96.8056476853 degree, input field ID=1.0 (if ID<0, use simple
        #   sinusoidal model, only works for the map integrator, phase is
        #   design phase with 0 for maximum energy gain), radius=1.0 m, x
        #   misalignment error=0.0m, y misalignment error=0.0m, rotation error
        #   x, y, z=0.0, 0.,
        #   0. Rad.
        "0.948049 80 1 104 34000000.0 650000000.0 96.8056476853 1 1.0 /",
    ],
    105: [
        # (use Fourier coefficients for Ez(z), not for map integrator).
        # length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF
        # frequency=700.0e6, driven phase=30.0 degree, input field ID=1.0,
        # radius=0.014m, x misalignment error=0.0m, y misalignment error=0.0m,
        # rotation error x, y, z=0.0, 0., 0. rad, Bz0=1.0 Tesla, 0. "aperture
        # size for wakefield", 0. "gap size for wk", 0. "length for wk". RF
        # structure wakefield only turned with length of wk>0.
        "1.48524 10 20 105 1.0 700.0e6 30. 1.0 0.014 0. 0. 0. 0. 0. 1. 0. 0. 0. /",
    ],
    106: [
        # (use Fourier coefficients for Ez(z), not for map integrator).
        # Traveling wave RF cavity, length=1.48524m, 10 "steps", 20 "map
        # steps", field scaling=1.0, RF frequency=700.0e6, driven phase=30.0
        # degree, input field ID=1.0, radius=0.014m, x misalignment error=0.0m,
        # y misalignment error=0.0m, rotation error x, y, z=0.0, 0., 0. rad,
        # (pi - beta * d) phase difference B and A, 0. "aperture size for
        # wakefield", 0. "gap size for wk", 0. "length for wk". RF structure
        # wakefield only turned with length of wk>0.
        "1.48524 10 20 106 1.0 700.0e6 30. 1.0 0.014 0. 0. 0. 0. 0. 0.5 0. 0. 0. /",
    ],
    110: [
        # length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF frequency=700.0e6, driven
        # phase=30.0 degree, input field ID=1.0, Xradius=0.014m, Yradius=0.014m, x misalignment
        # error=0.0m, y misalignment error=0.0m, rotation error x, y, z=0.0, 0., 0. rad), 1.0 (using discrete
        # data only, 2.0 using both discrete data and analytical function, other using analytical function only),
        # 2.0 (field in Cartesian coordinate, 1.0 in Cylindrical coordinate). The format of 3D field on Cartesian
        # grid is:
        # read(14,*,end=33)tmp1,tmp2,tmpint
        # XminRfg = tmp1
        # 6
        # XmaxRfg = tmp2
        # NxIntvRfg = tmpint !number of grid cells in x dimension
        # read(14,*,end=33)tmp1,tmp2,tmpint
        # YminRfg = tmp1
        # YmaxRfg = tmp2
        # NyIntvRfg = tmpint !number of grid cells in y dimension
        # read(14,*,end=33)tmp1,tmp2,tmpint
        # ZminRfg = tmp1
        # ZmaxRfg = tmp2
        # NzIntvRfg = tmpint !number of grid cells in z dimension
        # …
        # allocate(Exgrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # allocate(Eygrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # allocate(Ezgrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # allocate(Bxgrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # allocate(Bygrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # allocate(Bzgrid(NxIntvRfg+1,NyIntvRfg+1,NzIntvRfg+1))
        # …
        # read(14,*,end=77)tmp1,tmp2,tmp3,tmp4,tmp5,tmp6
        # n = n+1
        # k = (n-1)/((NxIntvRfg+1)*(NyIntvRfg+1))+1
        # j = (n-1-(k-1)*(NxIntvRfg+1)*(NyIntvRfg+1))/(NxIntvRfg+1) + 1
        # i = n - (k-1)*(NxIntvRfg+1)*(NyIntvRfg+1) - (j-1)*(NxIntvRfg+1)
        # Exgrid(i,j,k) = tmp1
        # Eygrid(i,j,k) = tmp2
        # Ezgrid(i,j,k) = tmp3
        # Bxgrid(i,j,k) = tmp4
        # Bygrid(i,j,k) = tmp5
        # Bzgrid(i,j,k) = tmp6
        "1.48524 10 20 110 1.0 700.0e6 30. 1.0 0.014 0.014 0. 0. 0. 0. 0. 1.0 2.0 /",
    ],
    -1: [
        "0. 0 0 -1 /",
    ],
    -2: [
        # NOTE: N must not equal 5 or 6 (Fortran code), or 24, 25, 26, 27, 29,
        # 30, 32. This prints the data set with 10 as sample frequency (i.e.
        # every 10 particles outputs 1 particle). Those particle data are
        # dimensionless in IMPACT internal unit. The normalization constants
        # are described in the following paritcle.in. If sample frequency is
        # negative, the output would be in the ImpactT particle format (except
        # that the longitudinal z is not absolute position but delta z, and pz
        # is gamma instead of gamma beta z).
        # "0. 0 N -2 0.0 10 /",
        "0. 0 1 -2 0.0 10 /",
        # "0. 0 N -2 0.0 10 /",  # TODO make it fail with N in {5, 6, 24, 25, 26, 27, 29, 30, 32}
    ],
    -3: [
        # radius=0.014m, xmax=0.02m,pxmax=0.02 mc,ymax=0.02m,pymax=0.02 mc,
        # zmax=0.02 7 rad,pzmax=0.02 mc^2, if no frame range is specified, the
        # program will use the maximum amplitude at given location as the
        # frame.
        "0. 0 0 -3 0.014 0.02 0.02 0.02 0.02 0.02 0.02 /",
    ],
    -4: [
        # radius=0.014m, xmax=0.02m,pxmax=0.02 mc,ymax=0.02m,pymax=0.02 mc,
        # zmax=0.02 rad,pzmax=0.02 mc^2, if no frame range is specified, the
        # program will use the maximum amplitude at given location as the
        # frame.
        "0. 0 0 -4 0.014 0.02 0.02 0.02 0.02 0.02 0.02 /",
    ],
    -5: [
        # radius=0.014m,xmax=0.02,pxmax=0.02mc,ymax=0.02,pymax=0.02mc,zmax=0.02rad,pzmax=0.0
        # 2 mc^2, if no frame range is specified, the program will use the
        # maximum amplitude at given location as the frame.
        "0. 0 0 -5 0.014 0.02 0.02 0.02 0.02 0.02 0.02 /",
    ],
    -6: [
        # radius=0.014m,xmax=0.02m,pxmax=0.02mc,ymax=0.02m,pymax=0.02mc,zmax=2degree,pzmax
        # =0.02 mc^2, if no frame range is specified, the program will use the
        # maximum amplitude at given location as the frame.)
        "0. 0 0 -6 0.014 0.02 0.02 0.02 0.02 2 0.02 /",
    ],
    -7: [
        # This function is used for restart purpose.
        "0. 0 1000 -7 /",
    ],
    -8: [
        # The fort.202 contains: bunch length coordinate (m), # of particles
        # per slice, current (A) per slice, X normalized emittance (m-rad) per
        # slice, Y normalized emittance (m-rad) per slice, dE/", E,
        # uncorrelated energy spread (eV) per slice, <x> (m) of each slice, <y>
        # (m) of each slice, X mismatch factor, Y mismatch factor. Here, the
        # Twiss parameters at that location are alphaX = 0.1, betaX = 1.0 (m),
        # alphaY=0.2, betaY = 2.0(m). If those parameters are not provided, the
        # mismatch factor should be neglected.
        "0.0 0 202 -8 101.0 0.1 1.0 0.2 2.0 /",
    ],
    -10: [
        # radius=0.014m (not used), 0.1 = xmis, 0.2 = pxmis, 0.3 = ymis, 0.4 =
        # pymis, 0.5 = tmis, 0.6 = ptmis.
        "0. 0 0 -10 0.014 0.1 0.2 0.3 0.4 0.5 0.6 /",
    ],
    -13: [
        # radius=0.014m (not used), xmin = -0.02m, xmax = 0.02m, ymin = -0.04m,
        # ymax = -0.04 0.04 /",
        # 0.04m.
        "0. 0 0 -13 0.014 -0.02 0.02 /",
    ],
    -14: [
        # toggle space charge
        "0.0 0 1 -14 0.1 -1.0 /",  # off
        "0.0 0 1 -14 0.1 1.0 /",  # on
    ],
    -18: [
        # radius=0.014m (not used), rotation angle = 0.5 rad.
        "0. 0 0 -18 0.014 0.5 /",
    ],
    -19: [
        # shift the beam longitudinally to the bunch centroid so that <dt>=<dE>=0. 8
        "0. 0 0 -19 /",
    ],
    -20: [
        # increased energy spread = 1000.0 eV. radius=0.014m (not used),
        "0. 0 0 -20 0.014 1000.0 /",  # TODO docs are wrong marking this as -18
    ],
    -21: [
        # radius=0.014m (not used),
        # xshift=0.02m,pxshift=0.02rad,yshift=0.02m,pyshift=0.02rad,
        # zshift=0.02deg,pzshift=0.02MeV.
        "0. 0 0 -21 0.014 0.02 0.02 0.02 0.02 0.02 0.02 /",
    ],
    -25: [
        # switch the integrator type using the “bmpstp” the 3rd number of line. hift the
        #             beam centroid in 6D phase space.
        # radius=0.01m (not used), use linear map integrator,
        "0. 0 1 -25 0.01 /",
        #             use the nonlinear Lorentz integrator for complicated external fields where transfer maps
        # are not available.
        "0. 0 2 -25 0.01 /",
    ],
    -40: [
        # radius=0.014m (not used), maximum voltage = 1.0e6 eV, -60 degrees,
        # and harmonic number (with respect to reference frequency) = 1.
        "0. 0 0 -40 0.014 1.e6 -60.0 1 /",
    ],
    -41: [
        # "1.0" not used, "41" is the file ID, "1.0" turn on or " -1.0" turn
        # off RF wakefield (if <10 no transverse wakefield effects included).
        "0.0 0 1 -41 1.0 41 1.0 /",
        "0.0 0 1 -41 1.0 41 -1.0 /",
    ],
    -52: [
        # uncorrelated energy spread by "7000.0" eV using laser wavelength
        # 1030d-9 m and the matched beam size "0.0895d-3" m)
        "0.0 0 10 -52 0.0895d-03 1030.0d-9 7000.0 /",
    ],
    -55: [
        # "1.0" not used, 0.0 – dipole (k0), 1.0 – quad. (k1), 2.0 – sext.
        # (k2), 3.0 – oct. (k3), 4.0 – dec. (k4), 5.0 – dodec. (k5).
        "0.0 0 10 -55 1.0 0.0 1.0 2.0 3.0 4.0 5.0 /",
    ],
    -99: [
        # This is useful if you have a big file want to run
        "0 1 1 -99 /",
    ],
}


def _make_pytest_params():
    for clsid, lines in doc_test_cases.items():
        ele_cls = input_element_by_id[clsid]
        for line in lines:
            yield pytest.param(
                ele_cls, line, id=f"id{clsid}_{ele_cls.__name__}_{line!r}"
            )


ele_doc_cases = pytest.mark.parametrize(
    ("ele_cls", "line"), list(_make_pytest_params())
)


@ele_doc_cases
def test_parse_element(ele_cls: type[InputElement], line: str):
    ele = InputElement.from_line(line)
    assert isinstance(ele, ele_cls)
    print(ele)


### Source code reference for manually verifying IMPACT-Z element parameters:
# Each of these parameters is 1-indexed *after* the type_id,
# so 'DriftTube' actually has 6 parameters: (length, steps, map_steps, type_id) and then (radius, )
"""
 DriftTube
radius  # 1

 Quadrupole
quad gradient  # 1
file ID  # 2
radius  # 3
x misalignment error  # 4
y misalignment error  # 5
rotation error x  # 6
rotation error y  # 7
rotation error z  # 8


 ConstFoc
x focusing gradient: kx0^2  # 1
y focusing gradient: ky0^2   # 2
z focusing gradient: kz0^2   # 3
radius   # 4


 Sol
Bz0  # 1
file ID  # 2
radius  # 3
x misalignment error  # 4
y misalignment error  # 5
rotation error x  # 6
rotation error y  # 7
rotation error z  # 8


 Dipole  TODO does not match
x field strength  # 1
y field strength  # 2
file ID: < 100, using t integration; > 100 but < 200 using z map + csr wake;  # 3
radius  # 4
! dx = this%Param(5)  - unused
! dy = this%Param(6)  - unused
! anglex = this%Param(7) - unused
! angley = this%Param(8) - unused
! anglez = this%Param(9) - unused
x misalignment error  # 10
y misalignment error  # 11
rotation error x  # 12
rotation error y  # 13
rotation error z  # 14


 DTL
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
quad 1 length  # 6
quad 1 gradient  # 7
quad 2 length  # 8
quad 2 gradient  # 9
x misalignment error for Quad 1  # 10
y misalignment error for Quad 1  # 11
rotation error x for Quad 1  # 12
rotation error y for Quad 1  # 13
rotation error z for Quad 1  # 14
x misalignment error for Quad 2  # 15
x misalignment error for Quad 2  # 16
rotation error x for Quad 2  # 17
rotation error y for Quad 2  # 18
rotation error z for Quad 2  # 19
x misalignment error for RF cavity  # 20
y misalignment error for RF cavity  # 21
rotation error x for RF cavity  # 22
rotation error y for RF cavity  # 23
rotation error z for RF cavity  # 24



 Multipole
id for sextupole(2), octupole(3), decapole(4)  # 1
field strength  # 2
file ID  # 3
radius  # 4
x misalignment error  # 5
y misalignment error  # 6
rotation error x  # 7
rotation error y  # 8
rotation error z  # 9


 CCDTL
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
x misalignment error  # 6
y misalignment error  # 7
rotation error x  # 8
rotation error y  # 9
rotation error z  # 10

 CCL
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
x misalignment error  # 6
y misalignment error  # 7
rotation error x  # 8
rotation error y  # 9
rotation error z  # 10


 SC
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
x misalignment error  # 6
y misalignment error  # 7
rotation error x  # 8
rotation error y  # 9
rotation error z  # 10

 SolRF
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
x misalignment error  # 6
y misalignment error  # 7
rotation error x  # 8
rotation error y  # 9
rotation error z  # 10
Bz0  # 11
aawk: aperture size for the wakefield  # 12
ggwk: gap size for the wakefield  # 13
lengwk: length for the wakefield  # 14



 TWS
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
radius  # 5
x misalignment error  # 6
y misalignment error  # 7
rotation error x  # 8
rotation error y  # 9
rotation error z  # 10
theta1 (pi - beta * d) phase difference B and A  # 11
aawk: aperture size for the wakefield   # 12
ggwk: gap size for the wakefield  # 13
lengwk: length for the wakefield  # 14


 EMfld
scale  # 1
RF frequency  # 2
theta0  # 3
file ID  # 4
x radius  # 5
y radius  # 6
x misalignment error  # 7
y misalignment error  # 8
rotation error x  # 9
rotation error y  # 10
rotation error z  # 11
flag for 3D discrete data(1),analytical+discrete(2),analytical(other)   # 12
flag for Cartisian(2) or Cylindrical coordintate(1)  # 13
"""
