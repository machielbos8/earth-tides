#--- General settings
verbose     no
loadtype    OTLregular
produce_csv yes

#--- Surfaceload
harmonic o1 k1 p1
surfaceload FES2014b
seawaterdensity 1030.0

#--- Describe methods of handling the surface load
extend  yes
extendmethod  average
conservewatermass  yes
conservemethod  Baker
preciseconservation  no
tidalinterpolationboundary  10.0
tidalinterpolation  Bos

#--- Where to find the list of stations
stationfile  Stations.dat

#--- Green's function and component
greensfunction  PREM.dat
greensinterpolation  spline
component  3

#--- Coastline
highrescoastlinefile quadtree_0.00549_v2.dat
highrescoastlineboundary  180.0
landwatermethod  exact
SPOTLAntarctica  no
usePolygons  no
MaskType 0

#--- Graphics
graphics no
plot_box no

#--- Mesh
generalmethod  Baker

#--- Cell refinement and what to do with cells over land
coastalblocks  halfremove
gridrefinement  yes
gridrefinementfactor  0.05
gridrefinementboundary  180.0
#--- CARGA
localphaseconvention  no
subtractreferencefield no
