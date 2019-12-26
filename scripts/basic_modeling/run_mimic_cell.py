"""
Test to run BD simulation for the mimic cell model including
granules and G-actins.
"""

import IMP
import IMP.atom
import IMP.rmf
import IMP.core
import IMP.algebra
import IMP.container
import RMF
import sys

#-----simulation parameters-----
print("Set parameters")
randomized_start = True
label = sys.argv[0]
L = 14000       # Size of box
R = 6000        # Radius of the cell
Rc = 4000       # Radius of the cortex
k = 10.0        # Strength of the harmonic boundary box
cutoff = 50.0
ngr = 100       # Number of granules
ndc = 5000      # Number of G-actin monomers in cortex layer

m = IMP.Model()
IMP.set_log_level(IMP.SILENT)
hc = IMP.atom.Hierarchy.setup_particle(m, IMP.Particle(m))

#-----read the initial system-----
print("Read the initial system")
rmf_file = RMF.open_rmf_file_read_only("cell_mimic.rmf")
hc = IMP.rmf.create_hierarchies(rmf_file, m)[0]
IMP.rmf.load_frame(rmf_file, RMF.FrameID(0))

#-----add rigid body restraints-----
# First restraints for granules and granule patches
print("----------")
print("Add rigid body restraints")
print("First for granules and patches")
hc_rb = [IMP.core.RigidBody.setup_particle(IMP.Particle(m, "GranuleRB_{}".format(i)), IMP.algebra.ReferenceFrame3D()) for i in range(0, ngr)]
for i in range(0, ngr):
    print("Granule_"+str(i))
    for root in hc.get_children():
        if root.get_name() == "Granule_" + str(i) or "GranulePatch_" + str(i) + "_" in root.get_name():
            hc_rb[i].set_coordinates_are_optimized(True)
            hc_rb[i].set_name("GranuleRB_" + str(i))
            hc_rb[i].add_member(root)
    rbd = IMP.atom.RigidBodyDiffusion.setup_particle(hc_rb[i])
    rbd.set_rotational_diffusion_coefficient(rbd.get_rotational_diffusion_coefficient() * ngr)
    rbd.set_coordinates_are_optimized(True)

# Then restraints for G-actins
print("Second for G-actins")
G_rb = [IMP.core.RigidBody.setup_particle(IMP.Particle(m, "G_actin_RB_{}".format(i)), IMP.algebra.ReferenceFrame3D()) for i in range(0, ndc)]
for i in range(0, ndc):
    print("G-actin monomer " + str(i))
    for root in hc.get_children():
        if root.get_name() == "monomer_" + str(i):
            G_rb[i].set_coordinates_are_optimized(True)
            G_rb[i].set_name("G_actin_RB_" + str(i))
            G_rb[i].add_member(root)
    Gbd = IMP.atom.RigidBodyDiffusion.setup_particle(G_rb[i])
    Gbd.set_rotational_diffusion_coefficient(Gbd.get_rotational_diffusion_coefficient() * ndc)
    Gbd.set_coordinates_are_optimized(True)

m.update()

#-----add harmonic bond-----
print("-----")
print("Add harmonic bond")
rs = []

#-----add excluded volume restraints-----
print("Add excluded volume restraints")
ev = IMP.core.ExcludedVolumeRestraint(IMP.atom.get_leaves(hc), 1, 10, "EV")
rs.append(ev)

#-----bounding box-----
print("Add bounding box")
# Outer bounding box
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-L/2, -L/2, -L/2), IMP.algebra.Vector3D(L/2, L/2, L/2))

# PBC bounding sphere
pbc_sphere = IMP.algebra.Sphere3D([0,0,0], R)

# Cortex layer sphere
cortex_sphere = IMP.algebra.Sphere3D([0,0,0], Rc)

# Enclosing spheres for entire contents
print("Enclosing spheres for entire contents")
pbc_bsss = IMP.core.BoundingSphere3DSingletonScore(IMP.core.HarmonicUpperBound(0, k), pbc_sphere)
cortex_bsss = IMP.core.BoundingSphere3DSingletonScore(IMP.core.HarmonicUpperBound(0, k), cortex_sphere)
outer_bbss = IMP.core.BoundingBox3DSingletonScore(IMP.core.HarmonicUpperBound(0, k), bb)
rs.append(IMP.container.SingletonsRestraint(pbc_bsss, hc.get_children()))
rs.append(IMP.container.SingletonsRestraint(cortex_bsss, hc.get_children()))
rs.append(IMP.container.SingletonsRestraint(outer_bbss, hc.get_children()))

print("Preparing scoring functions")
sf = IMP.core.RestraintsScoringFunction(rs, "SF")
for yc in hc.get_children():
    rb = IMP.atom.RigidBodyDiffusion(yc)
    rb.set_coordinates_are_optimized(True)

m.update()

#-----run-----
print("run")
frames = 10000

bd = IMP.atom.BrownianDynamics(m)
bd.set_log_level(IMP.SILENT)
bd.set_scoring_function(sf)
bd.set_maximum_time_step(10000)
bd.set_temperature(310)

rmf = RMF.create_rmf_file("mimic_cell_sim.rmf")
rmf.set_description("Brownian dynamics trajectory with 10fs timestep.\n")
IMP.rmf.add_hierarchy(rmf, hc)
IMP.rmf.add_hierarchy(rmf, rs)
IMP.rmf.add_geometry(rmf, IMP.display.BoundingBoxGeometry(bb))
IMP.rmf.add_geometry(rmf, IMP.display.SphereGeometry(pbc_sphere))
IMP.rmf.add_geometry(rmf, IMP.display.SphereGeometry(cortex_sphere))

# optimizer state
os = IMP.rmf.SaveOptimizerState(m, rmf)
os.update_always("initial conformation")
os.set_log_level(IMP.SILENT)
os.set_simulator(bd)
os.set_period(1000)
bd.add_optimizer_state(os)
bd.optimize(frames)

