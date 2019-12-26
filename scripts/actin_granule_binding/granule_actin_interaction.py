"""
This script is written for generating one filament polymerized by actin 
monomers and one granule. Granule's binding and movement along filament
will be modelled.
The original two scripts, mimic_filament.py and run_mimic_filament.py 
will be combined.

This script is the updated version from multiple_filament_model.py.
1. Excluded volumes and Bipartite containers have been updated to latest version.
2. Only one granule and one filament are generated for testing the interaction.

Addition:
    Use IMP.npctransport.FunctorLinearInteractionPairScore
"""

from __future__ import print_function, division
import IMP
import IMP.atom
import IMP.algebra
import IMP.core
import IMP.container
import IMP.display
import IMP.rmf
import IMP.npc
import IMP.npctransport
import sys
import RMF
import re
import numpy as np
import math
import random


# Function for rotating vectors.
# This will be used in setting new binding patches from existing patches.
def set_third_patch(vl, vr, R):
    direction = np.cross(vr, vl - vr)
    ld = np.linalg.norm(direction)
    new_d = vr + direction * R / ld
    return new_d

def set_fourth_patch(v3, vr):
    v4 = vr + (vr - v3)
    return v4

# Function to set parameters of IMP particles
def set_param_for_particle(model, p, v, R, mass, c):
    a = IMP.core.XYZR.setup_particle(p)
    a.set_coordinates_are_optimized(True)
    a.set_coordinates(v)
    a.set_radius(R)
    IMP.atom.Mass.setup_particle(a, mass)
    IMP.atom.Diffusion.setup_particle(model, a)
    IMP.display.Colored.setup_particle(a, IMP.display.get_display_color(c))
    
    return a

# Function for converting real time to frames in simulating step
# Time unit is nanosecond (ns)
def convert_time_ns_to_frames(time_ns, step_size_fs):
    FS_PER_NS = 1E6
    time_fs = time_ns * FS_PER_NS
    n_frames_float = (time_fs + 0.0) / step_size_fs
    n_frames = int(round(n_frames_float))
    return max(n_frames, 1)



# Function for generating the first monomer sphere for each filament.
def create_starting_monomer(model, vec, R, mass):
    p = IMP.Particle(model, "Actin_0")
    a = set_param_for_particle(model, p, vec, R, mass, 0)

    return a


# Function for generating binding patches for the first monomer
# Four patches will be created. Two among them will be used in
# connection with neighbouring actin monomers.
def create_patch_for_first_monomer(model, a, R, r):
    c = IMP.algebra.Vector3D(a.get_coordinates())
    sp = IMP.algebra.Sphere3D(c, R)
#    lt, rt = IMP.algebra.get_uniform_surface_cover(sp, 2)
    lt = IMP.algebra.get_random_vector_on(sp)
    rt = lt + 2*(c - lt)
    left = IMP.algebra.Vector3D(lt)
    right = IMP.algebra.Vector3D(rt)
# Actin binding patch: left
    l0 = IMP.Particle(model, "ActinPatches_0_0")
    apl = set_param_for_particle(model, l0, left, r, 1, 1)
# Actin binding patch: right
    r0 = IMP.Particle(model, "ActinPatches_0_1")
    apr = set_param_for_particle(model, r0, right, r, 1, 1)
# To make the next monomer linked directly to one of the patches, get the
# coordinate of connecting point on patch's sphere
    fr = right + (right - c)*r/R
    far_right = IMP.algebra.Vector3D(fr)
# New contents added: generating another two binding patches
# Add the third binding patch
    up = set_third_patch(left, c, R)
    u0 = IMP.Particle(model, "ActinPatches_0_2")
    apu = set_param_for_particle(model, u0, up, r, 1, 1)
# Add the fourth binding patch
    down = set_fourth_patch(up, c)
    d0 = IMP.Particle(model, "ActinPatches_0_3")
    apd = set_param_for_particle(model, d0, down, r, 1, 1)

    return apl, apr, apu, apd, right, far_right

# Function for generating monomers attached to previous one
# The filament will elongate with number of monomers growing
# Six parameters are required:
# 1. model for IMP.Model()
# 2. i for number of the monomer
# 3. ri for right, coordinate of the previous right patch's center
# 4. fr for far right, coordinate of the connecting point at the 
#    surface of the previous right patch
# 5. R for radius of the actin monomer
# 6. r for radius of the patch

def create_new_monomer(model, i, ri, fr, R, r, mass):
# Set the information of monomer
    p = IMP.Particle(model, "Actin_{}".format(i))
# Set the information of four patches
    p1 = IMP.Particle(model, "ActinPatches_"+str(i)+"_0")
    p2 = IMP.Particle(model, "ActinPatches_"+str(i)+"_1")
    p3 = IMP.Particle(model, "ActinPatches_"+str(i)+"_2")
    p4 = IMP.Particle(model, "ActinPatches_"+str(i)+"_3")
# Calculate coordinates for the centers of monomer and patches
    lt = fr + (fr - ri)
    left = IMP.algebra.Vector3D(lt)
    c = lt + (lt - fr) * R / r
    center = IMP.algebra.Vector3D(c)
    rt = c + (c - lt)
    right = IMP.algebra.Vector3D(rt)
    fr1 = rt + (rt - c) * r / R
    far_right = IMP.algebra.Vector3D(fr1)
    up = set_third_patch(left, c, R)
    down = set_fourth_patch(up, c)
# Define the mass, diffusion coefficients, etc. for monomer
    a = set_param_for_particle(model, p, center, R, mass, 0)
# Define the mass, diffusion coefficients, etc. for left patch
    apl = set_param_for_particle(model, p1, left, r, mass, 0)
# Define the mass, diffusion coefficients, etc. for right patch
    apr = set_param_for_particle(model, p2, right, r, mass, 0)
# Define the mass, diffusion coefficients, etc. for up patch
    apu = set_param_for_particle(model, p3, up, r, mass, 0)
# Define the mass, diffusion coefficients, etc. for down patch
    apd = set_param_for_particle(model, p4, down, r, mass, 0)

    return a, apl, apr, apu, apd, right, far_right


# Function for creating granule with binding patches
def create_new_granule(model, i, v, R, r, mass):
# Set the information of a granule
    p = IMP.Particle(model, "Granule_{}".format(i))
    a = set_param_for_particle(model, p, v, R, mass, 2)
# Generate six binding patches for the granule
    sp = IMP.algebra.Sphere3D(v, R)
    patch_set = [IMP.core.XYZR.setup_particle(IMP.Particle(model, "GranulePatch_" + str(i) + "_{}".format(j))) for j in range(6)]
    for j in range(6):
        patch_set[j].set_coordinates(IMP.algebra.get_uniform_surface_cover(sp, 6)[j])
        patch_set[j].set_coordinates_are_optimized(True)
        patch_set[j].set_radius(3)
        IMP.atom.Mass.setup_particle(patch_set[j], 1)
        IMP.atom.Diffusion.setup_particle(model, patch_set[j])
        IMP.display.Colored.setup_particle(patch_set[j], IMP.display.get_display_color(0))
    
    return a, patch_set

    

#------------
#------------
# Main block
#------------
#------------

#--------
# Generate the modeling environment
#--------
# The Boundary Box
L = 350
bb = IMP.algebra.BoundingBox3D(IMP.algebra.Vector3D(-L/2, -L/2, -L/2), IMP.algebra.Vector3D(L/2, L/2, L/2))

# PBC bounding sphere
R = 180
pbc_sphere = IMP.algebra.Sphere3D([0,0,0], R)

# Set a range to prevent generating monomers outside the bounding sphere
barrier = 170
barrier_sphere = IMP.algebra.Sphere3D([0,0,0], barrier)

# Generate the contents of the cell model
m = IMP.Model()

#-------
# 1. Nucleus
#-------
h_root = IMP.atom.Hierarchy.setup_particle(IMP.Particle(m, "root"))
# In testing period, we do not need the nuclues.
# So the code will be added later.

#-------
# 2. Granule
#-------
# Set the radius for actin monomer, granule and patches
bead_radius = 5
granule_radius = 30
patch_radius = 0.5
# Set mass for granule and actin
actin_mass = 105000
# granule mass is estimated and waiting to be replaced by
# experimental data from references
granule_mass = 1E7

# Total number of granules
ng = 1
# Initialize lists for granules and patches
granules = []
granule_patches = []
# Initialize list to set rigidbody restraints for each granule
granule_rb = [IMP.core.RigidBody.setup_particle(IMP.Particle(m, "GranuleRB_"+str(i)), IMP.algebra.ReferenceFrame3D()) for i in range(ng)]

# Generate the granule and patches
for i in range(ng):
    g_coordinate = IMP.algebra.get_random_vector_in(barrier_sphere)
    gra, gra_p = create_new_granule(m, i, g_coordinate, granule_radius, patch_radius, granule_mass)
    granules.append(gra)
    granule_patches.append(gra_p)
# Set the granule and patches as an entire rigid body
    granule_rb[i].set_coordinates_are_optimized(True)
    granule_rb[i].set_name("Granule_"+str(i))
    granule_rb[i].add_member(gra)
    for ele in gra_p:
        granule_rb[i].add_member(ele)
# Add the granule and patches to hierarchy tree
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(gra))
    for ele in gra_p:
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(ele))
# Set diffusion coefficients
    g_rbd = IMP.atom.RigidBodyDiffusion.setup_particle(granule_rb[i])
    D_E = IMP.atom.get_einstein_diffusion_coefficient(granule_radius) * 5
    g_rbd.set_diffusion_coefficient(D_E)
    D_R = IMP.atom.get_einstein_rotational_diffusion_coefficient(granule_radius) * 5
    g_rbd.set_rotational_diffusion_coefficient(D_R)
    g_rbd.set_coordinates_are_optimized(True)

# Output all granules
# print("Output all granules")
# for i in range(ng):
#     print(granules[i])
#     for j in range(6):
#         print(granule_patches[i])


#-------
# 3. Filaments
#-------
# Total number of filaments
nf = 15
# Number of actins in each filament
nam = 12
# Get the initiate coordinate for the first monomers
initiate = []
for i in range(nf):
    vi = IMP.algebra.get_random_vector_in(barrier_sphere)
    initiate.append(vi)

# Initialize two-dimension lists to restore the actin monomers
# and binding patches
actins = []
actin_patches = []
for i in range(nf):
    actins.append([])
    actin_patches.append([])

# Initialize list to set rigidbody restraints of F-actin
filament_rb = [IMP.core.RigidBody.setup_particle(IMP.Particle(m, "F-actin_RB_"+str(i)), IMP.algebra.ReferenceFrame3D()) for i in range(nf)]

# Generate the first monomers for each filament
for i in range(nf):
    init = create_starting_monomer(m, initiate[i], bead_radius, actin_mass)
    actins[i].append(init)
# Ready to set rigidbody restraints for each filament
    filament_rb[i].set_name("F-actin_"+str(i))
    filament_rb[i].add_member(init)
    filament_rb[i].set_coordinates_are_optimized(True)
# Add actin to hierarchy tree
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(init))

# Initialize lists to contain the four patches on each monomer
first_left = []
first_right = []
first_far_right = []
first_up = []
first_down = []
filament_lengths = []
# Generate the patches
for i in range(nf):
    apl, apr, apu, apd, right, far_right = create_patch_for_first_monomer(m, actins[i][0], bead_radius, patch_radius)
    first_left.append(apl)
    first_right.append(apr)
    first_far_right.append(far_right)
    first_up.append(apu)
    first_down.append(apd)
    actin_patches[i].append(apl)
    actin_patches[i].append(apr)
    actin_patches[i].append(apu)
    actin_patches[i].append(apd)
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(apl))
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(apr))
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(apu))
    h_root.add_child(IMP.atom.Hierarchy.setup_particle(apd))
# Add these patches to rigid body
    filament_rb[i].add_member(apl)
    filament_rb[i].add_member(apr)
    filament_rb[i].add_member(apu)
    filament_rb[i].add_member(apd)

# print(first_right)

# Elongation of each filament
for i in range(nf):
    right = first_right[i].get_coordinates()
    far_right = first_far_right[i]
    for j in range(1, nam):
        a, apl, apr, apu, apd, right, far_right = create_new_monomer(m, j, right, far_right, bead_radius, patch_radius, actin_mass)
        actins[i].append(a)
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(a))
        actin_patches[i].append(apl)
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(apl))
        actin_patches[i].append(apr)
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(apr))
        actin_patches[i].append(apu)
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(apu))
        actin_patches[i].append(apd)
        h_root.add_child(IMP.atom.Hierarchy.setup_particle(apd))
# Add new actins and patches to rigid body
        filament_rb[i].add_member(a)
        filament_rb[i].add_member(apl)
        filament_rb[i].add_member(apr)
        filament_rb[i].add_member(apu)
        filament_rb[i].add_member(apd)
# Terminate elongate if the filament has extended outside the sphere
        c = IMP.algebra.Vector3D(a.get_coordinates())
        length = c.get_magnitude()
        if length > barrier:
            break
# Set diffusion coefficients after elongation finishes
## Remove F-actin diffusion for testing
#    f_rbd = IMP.atom.RigidBodyDiffusion.setup_particle(filament_rb[i])
#    D_E = IMP.atom.get_einstein_diffusion_coefficient(bead_radius) * 0.1
#    f_rbd.set_diffusion_coefficient(D_E)
#    D_R = IMP.atom.get_einstein_rotational_diffusion_coefficient(bead_radius) * 0.1
#    f_rbd.set_rotational_diffusion_coefficient(D_R)
    
# Output the first F-actin for testing
# print("Output the first F-actin for testing")
# for ele in actins[0]:
#     print(ele)



print("Total number of actin monomers:")
total_num = i
print(total_num)


#----------
# Modeling
#----------
# Create an RMF file to save the model
sa = RMF.create_rmf_file("Test_multiply_granules_and_filaments.rmf")
sa.set_description("Model of granule and actin filament")
IMP.rmf.add_hierarchy(sa, h_root)
IMP.rmf.save_frame(sa, "0")

# Set parameters for modeling
print("Set parameters")
randomized_start = True
label = sys.argv[0]
kh = 0.1 # Strength of the harmonic boundary box in kcal/mol/A^2
k_excluded = 0.1 # Strength of lower-harmonic excluded volume score in kcal/mol/A^2
k_patches = 0.2 # Strength of patch interaction in kcal/mol/A^2
k_bipartite_pair = 0.2 # Strength of interaction between actin patches and granule patches

IMP.set_log_level(IMP.SILENT)

# Set restraints for the system
# List for containing the restraints
rs = []

# 1. Restraints for specific interaction between patches
print("-----")
print("Set restraints between patches")

# Use functions in IMP.npctransport module
range_patches_A = 3.0 # Range of patch interactions in angstroms


# Pick out all the patches in the system
# Warning: Only the interactions between granule and actins will be
# calculated. Interaction between actins should be neglected.

# binding_patches = [[] for i in range(ng)]
binding_patches_g = []
binding_patches_a = []
for root in h_root.get_children():
    if "GranulePatch_" in root.get_name():
        binding_patches_g.append(root.get_particle())
    elif "ActinPatches_" in root.get_name() and "_2" in root.get_name():
        binding_patches_a.append(root.get_particle())
    elif "ActinPatches_" in root.get_name() and "_3" in root.get_name():
        binding_patches_a.append(root.get_particle())

print(binding_patches_g[0])
print(binding_patches_a[0])

# Create the container
# apgp: actin patch & granule patch
bcpc_apgp = IMP.container.CloseBipartitePairContainer(binding_patches_g, binding_patches_a, range_patches_A, k_bipartite_pair)
# Old scoring function:
# Use Linear interaction pair score in npctransport module
# lips = IMP.npctransport.LinearInteractionPairScore(k_excluded, range_patches_A, k_patches)

# Use IMP.npctransport.LinearInteractionPairScore
# Variables required are: k_replusive, attraction_range, k_attractive
# 1. k_repulsive: two sphere particles will repulse each other when distance is smaller than 0;
# 2. attraction_range: distance range in which two particles attract each other;
# 3. k_attractive;
# Notice: since the granules will bind to F-actin, k_repulsive will not be set large.
k_rep = 0.01
a_range = 10.0
k_att = 10.0
flips_pp = IMP.npctransport.FunctorLinearInteractionPairScore(k_rep, a_range, k_att)

# Set score over dynamic container
pr_pp = IMP.container.PairsRestraint(flips_pp, bcpc_apgp)
rs.append(pr_pp)

m.update()


# 2. Restraints for non-specific interactions between actin binding patch and granule sphere
granule_sphere = []
# Get granules without binding patches
for root in h_root.get_children():
    if "Granule_" in root.get_name():
        granule_sphere.append(root.get_particle())
# Create the container
# apgs: actin patch & granule sphere
bcpc_apgs = IMP.container.CloseBipartitePairContainer(granule_sphere, binding_patches_a, range_patches_A, k_bipartite_pair)
# Scoring function
# Parameters for non-specific interaction
k_rep_n = 0.1
a_range_n = 5.0
k_att_n = 0.1
flips_ps = IMP.npctransport.FunctorLinearInteractionPairScore(k_rep_n, a_range_n, k_att_n)
# Set score over dynamic container
pr_ps = IMP.container.PairsRestraint(flips_ps, bcpc_apgs)
rs.append(pr_ps)

m.update()



# 3. Add excluded volume restraints
print("-----")
print("Add excluded volume restraints")
# Slack: how far in A the particles must move before the internal list of close pairs is computed.
slack_ev = 5.0
# Only granules and actins will be added with excluded volume restraints.
# Binding patches will not exclude when they interact.
main_particles = []
for root in h_root.get_children():
    if "Actin_" in root.get_name() or "Granule_" in root.get_name():
        main_particles.append(root)
nbl = IMP.container.ClosePairContainer(main_particles, 0, k_excluded)
# Use functions from core/excluded_volume.py
hlb = IMP.core.HarmonicLowerBound(0, k_excluded)
sd = IMP.core.SphereDistancePairScore(hlb)
# Use the lower bound on the inter-sphere distance to push spheres apart
nbr = IMP.container.PairsRestraint(sd, nbl)
rs.append(nbr)
# ev = IMP.core.ExcludedVolumeRestraint(total_particles, k_excluded, slack_ev, "EV")
# rs.append(ev)

m.update()




# Bounding Box
print("Add bounding box")
# Enclosing spheres for entire contents
print("Enclosing spheres for entire contents")
singleton_score_pbc = IMP.core.HarmonicUpperBound(0, kh)
pbc_bsss = IMP.core.BoundingSphere3DSingletonScore(singleton_score_pbc, pbc_sphere)
singleton_score_outer = IMP.core.HarmonicUpperBound(0, kh)
outer_bbss = IMP.core.BoundingBox3DSingletonScore(singleton_score_outer, bb)

singleton_restraint_pbc = IMP.container.SingletonsRestraint(pbc_bsss, h_root.get_children())
rs.append(singleton_restraint_pbc)
singleton_restraint_outer = IMP.container.SingletonsRestraint(outer_bbss, h_root.get_children())
rs.append(singleton_restraint_outer)

# print("Check the hierarchy before setting scoring function.")
# print(h_root.get_children())


# Set the scoring function
print("Preparing scoring function")
sf = IMP.core.RestraintsScoringFunction(rs, "SF")
for yc in h_root.get_children():
    rb = IMP.atom.RigidBodyDiffusion(yc)
    rb.set_coordinates_are_optimized(True)

m.update()


# Set the time parameter
BD_STEP_SIZE_SEC = 1E-12
SIM_TIME_SEC = 1E-9
bd_step_size_fs = BD_STEP_SIZE_SEC * 1E15
sim_time_ns = SIM_TIME_SEC * 1E9
RMF_DUMP_INTERVAL_NS = sim_time_ns / 1000.0
sim_time_frames = convert_time_ns_to_frames(sim_time_ns, bd_step_size_fs)
rmf_dump_interval_frames = convert_time_ns_to_frames(RMF_DUMP_INTERVAL_NS, bd_step_size_fs)
print("Simulation time " + str(sim_time_ns) + " / " + str(sim_time_frames) + " frames.")
print("RMF dump interval " + str(RMF_DUMP_INTERVAL_NS) + " / " + str(rmf_dump_interval_frames) + " frames.")


# Running
print("run")

bd = IMP.atom.BrownianDynamics(m)
bd.set_log_level(IMP.SILENT)
#bd.set_log_level(IMP.VERBOSE)
bd.set_scoring_function(sf)
bd.set_maximum_time_step(bd_step_size_fs)
bd.set_temperature(310)

rmf = RMF.create_rmf_file("test_granule_actin_interaction_functor.rmf")
rmf.set_description("Brownian dynamics trajectory with" + str(bd_step_size_fs) +"fs timestep.\n")
IMP.rmf.add_hierarchy(rmf, h_root)
IMP.rmf.add_restraints(rmf, rs)
IMP.rmf.add_geometry(rmf, IMP.display.BoundingBoxGeometry(bb))
IMP.rmf.add_geometry(rmf, IMP.display.SphereGeometry(pbc_sphere))
# IMP.rmf.add_geometry(rmf, IMP.display.SphereGeometry(cortex_sphere))

# optimizer state
os = IMP.rmf.SaveOptimizerState(m, rmf)
os.update_always("initial conformation")
os.set_log_level(IMP.SILENT)
#os.set_log_level(IMP.VERBOSE)
os.set_simulator(bd)
os.set_period(rmf_dump_interval_frames)
bd.add_optimizer_state(os)
bd.optimize(sim_time_frames)

