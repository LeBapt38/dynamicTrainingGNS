-- mpm 3d
math.randomseed(123)

--numerical parameters
end_frame = 80
dx = 0.01
gravity = TV.create({0, -9.81, 0})
newton_iterations = 100
quasistatic = false
symplectic = true
frame_dt= 1/100
matrix_free = true
verbose = false
cfl = 0.6

use_particle_collision = false

transfer_scheme = FLIP_blend_PIC 
flip_pic_ratio = 0.99

--physical prameters
a = math.random()
b = math.random()
c = math.random()
d = math.random()
randomness = 0
Youngs = 300000.0
ryoung = Youngs * (1 + ((0.5 - a) * randomness))
nu = 0.3
rnu = nu * (1 + ((0.5 - b) * randomness))
rho = 25000
rrho = rho * (1 + ((0.5 - c) * randomness))

friction_angle =23
rfriction_angle = friction_angle * (1 + ((0.5 - d) * randomness))

volFriction = 0.31

max_dt = 0.6 * dx * math.sqrt(rrho/ryoung)

i = 0
output = "/media/user/Volume/granular_collapse_GNS_dyn/train/300000_30_25000_23_halfPlane_0-1"
function initialize(frame)
	--ici on charge la géométrie
	local min_corner = TV.create({-0.0705,0, -0.0705})
	local max_corner = TV.create({.0705,.0564, .0705})
	local box = AxisAlignedAnalyticBox.new(min_corner, max_corner)
	local particles_hande = mpm:sampleInAnalyticLevelSet(box, rrho, 8)

	--ici on charge la loi elastique
	local m = StvkWithHencky.new(ryoung,rnu)
	particles_hande:addFBasedMpmForce(m)

	--ici on charge la loi constitutive
	local p = DruckerPragerStvkHencky.new(friction_angle)
	particles_hande:addPlasticity(m, p, "F")


	-- on charge une condition aux limites
	local ground_origin = TV.create({-0.5,0.0, 0.0})
	local ground_normal = TV.create({0,1,0})
	local ground_ls = HalfSpace.new(ground_origin, ground_normal)
	local ground_object = AnalyticCollisionObject.new(ground_ls, SEPARATE)
	ground_object:setFriction(volFriction)
	mpm:addAnalyticCollisionObject(ground_object)

end

