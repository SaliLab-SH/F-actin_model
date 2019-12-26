"""
Testing the layer model together with granules
Sizes of contents are not directly taken from experiments and refs.
This code is used only for testing the methods.
"""
from __future__ import print_function, division
import IMP
import IMP.atom
import IMP.algebra
import IMP.rmf
import RMF
import IMP.container
import IMP.display
import sys
from math import sqrt

#------boundary box-----
# Outer box
L = 14000
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-L/2,-L/2,-L/2), IMP.algebra.Vector3D(L/2,L/2,L/2))

# PBC bounding sphere
R = 6000
pbc_sphere = IMP.algebra.Sphere3D([0,0,0], R)

#-----cell cortex-----
# Cell cortex's thickness is usually less than 0.1 um.
Rc = 4000
cortex_sphere = IMP.algebra.Sphere3D([0,0,0], Rc)

#-----nucleus-----
m = IMP.Model()
h_root = IMP.atom.Hierarchy.setup_particle(IMP.Particle(m, "root"))
ne = IMP.core.XYZR.setup_particle(IMP.Particle(m, "nucleu"))
ne.set_coordinates_are_optimized(True)
ne.set_coordinates([0,0,0])
ne.set_radius(1000)
IMP.atom.Mass.setup_particle(ne, 1)
IMP.display.Colored.setup_particle(ne, IMP.display.get_display_color(2))
h_root.add_child(IMP.atom.Hierarchy.setup_particle(ne))

#-----cortex F-actin meshwork-----
# Add contents into the cortex layer
# It has to be discussed whether the layer should be made of continuous
# contents or discrete contents representing actin monomers

# First, discrete contents are tested

ndc = 5000

# Prepare a list of coordinates in the area between cell sphere and 
# cortex sphere.
coor_cont = []
lcn = 0
while lcn <= ndc:
    coor = IMP.algebra.get_random_vector_in(pbc_sphere)
    ra = sqrt(coor[0]**2 + coor[1]**2 + coor[2]**2)
    if ra <= Rc:
        continue
    else:
        coor_cont.append(coor)
        lcn = len(coor_cont)


mono_test = [IMP.core.XYZR.setup_particle(IMP.Particle(m, "monomer_{}".format(i))) for i in range(0, ndc)]
for i in range(0, len(mono_test)):
    mono_test[i].set_coordinates_are_optimized(True)
# The content must be generated in the area between cell sphere and 
# cortex sphere.
    mono_test[i].set_coordinates(coor_cont[i])
    mono_test[i].set_radius(100)
    IMP.atom.Mass.setup_particle(mono_test[i], 1)
    IMP.atom.Diffusion.setup_particle(m, mono_test[i])
    IMP.display.Colored.setup_particle(mono_test[i], IMP.display.get_display_color(3))
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(mono_test[i]))


#-----Granules-----
ngr = 100
granule = [IMP.core.XYZR.setup_particle(IMP.Particle(m, "Granule_{}".format(i))) for i in range(0, ngr)]
granule_patch = [IMP.core.XYZR.setup_particle(IMP.Particle(m, "Granule_{}".format(i))) for i in range(0, 6*ngr)]
for i in range(0, len(granule)):
    granule[i].set_coordinates_are_optimized(True)
    granule[i].set_coordinates(IMP.algebra.get_random_vector_in(cortex_sphere))
    granule[i].set_radius(350)
    IMP.atom.Mass.setup_particle(granule[i], 1)
    IMP.atom.Diffusion.setup_particle(m, granule[i])
    IMP.display.Colored.setup_particle(granule[i], IMP.display.get_display_color(0))
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(granule[i]))
# add patches for each granule
    granule_patch[i] = [IMP.core.XYZR.setup_particle(IMP.Particle(m, "GranulePatch_" + str(i) + "_{}".format(j))) for j in range(0, 6)]
    for j in range(0, 6):
        granule_patch[i][j].set_coordinates(IMP.algebra.get_uniform_surface_cover(IMP.algebra.Sphere3D(IMP.algebra.Vector3D(granule[i].get_coordinates()), 350), 6)[j])
        granule_patch[i][j].set_coordinates_are_optimized(True)
        granule_patch[i][j].set_radius(10)
        IMP.atom.Mass.setup_particle(granule_patch[i][j], 1)
        IMP.atom.Diffusion.setup_particle(m, granule_patch[i][j])
        IMP.display.Colored.setup_particle(granule_patch[i][j], IMP.display.get_display_color(0))
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(granule_patch[i][j]))


#-----write output-----
out = RMF.create_rmf_file("cell_mimic.rmf")
IMP.rmf.add_hierarchy(out, h_root)
IMP.rmf.add_geometry(out, IMP.display.BoundingBoxGeometry(bb))
IMP.rmf.add_geometry(out, IMP.display.SphereGeometry(pbc_sphere))
IMP.rmf.add_geometry(out, IMP.display.SphereGeometry(cortex_sphere))
IMP.rmf.save_frame(out, "0")
