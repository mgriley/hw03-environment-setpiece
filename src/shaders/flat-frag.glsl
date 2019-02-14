#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

uniform bvec3 u_bin;
uniform vec3 u_fin;

in vec2 fs_Pos;
out vec4 out_Col;

const float pi = 3.14159;
const float v_fov = pi / 4.0;

struct AABB {
  vec3 min;
  vec3 max;
};

// Noise functions

vec2 hash2( vec2 p ) { p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*18.5453); }
vec3 hash3( float n ) { return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(338.5453123,278.1459123,191.1234)); }
float hash(vec2 p) {
	return fract(dot(hash2(p),vec2(1.0,0.0)));
}
vec3 hash3(vec3 p) {
	p=vec3(dot(p,vec3(127.1,311.7,732.1)),dot(p,vec3(269.5,183.3,23.1)),dot(p,vec3(893.1,21.4,781.2))); return fract(sin(p)*18.5453);	
}
float hash3to1(vec3 p) {
	return fract(dot(hash3(p),vec3(32.32,321.3,123.2)));
}

float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float random1( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

vec2 random2( vec2 p , vec2 seed) {
  return fract(sin(vec2(dot(p + seed, vec2(311.7, 127.1)), dot(p + seed, vec2(269.5, 183.3)))) * 85734.3545);
}

float surflet_noise(vec2 p, vec2 seed) {
  // use the surface-lets technique
  // scale is the length of a cell in the perlin grid
  float scale = 10.0;
  vec2 base = floor(p / scale);
  vec2 corners[4] = vec2[4](
    base,
    base + vec2(1.0, 0.0),
    base + vec2(0.0, 1.0),
    base + vec2(1.0, 1.0)
  );
  float sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    // this gives decent noise, too
    /*
    vec2 corner = scale * corners[i];
    float corner_h = random1(corner, seed);
    vec2 delta = corner - p;
    float weight = 1.0 - smoothstep(0.0, scale, length(delta));
    sum += weight * corner_h;
    */

    vec2 corner = scale * corners[i];
    vec2 corner_dir = 2.0 * random2(corner, seed) - vec2(1.0);
    vec2 delta = p - corner;
    // this is the height if we were only on a slope of
    // magnitude length(corner_dir) in the direction of corner_dir
    float sloped_height = dot(delta, corner_dir);
    float weight = 1.0 - smoothstep(0.0, scale, length(delta));
    sum += 0.25 * weight * sloped_height;
  }
  return (sum + 1.0) / 2.0;
}

float some_noise(vec2 p, vec2 seed) {
  float noise = surflet_noise(p, seed);
  return 10.0 * (noise - 0.5) * 2.0;
}

float fbm_noise(vec2 p, vec2 seed) {
  // Note: using surflet_noise makes this slowww
  float sum = 0.0;
  float persistence = 0.5;
  for (int i = 0; i < 2; ++i) {
    float amp = pow(persistence, float(i));
    float freq = pow(2.0, float(i));
    sum += surflet_noise(p * freq, seed) * amp;
  }
  return sum;
}

float smooth_grad_2d(vec2 pos) {
	vec2 g = floor(pos);
	vec2 f = fract(pos);

	vec2 b = vec2(0.0,1.0);
	vec2 points[4] = vec2[4](b.xx, b.xy, b.yx, b.yy);
	float sum = 0.0;
	for (int i = 0; i < points.length(); ++i) {
		vec2 grad = 2.0*(hash2(g+points[i])-0.5);
			vec2 delta = f-points[i];
			float weight = 1.0-smoothstep(0.0,1.0,length(delta));
			sum += weight*dot(grad,delta);
	}
	
	return clamp(0.0,1.0,0.5+0.5*sum);
}

float gradient_noise_2d(vec2 pos) {
	vec2 g = floor(pos);
	vec2 f = fract(pos);
	vec2 inter = f*f*f*f*(f*(f*6.0-15.0)+10.0);

	vec2 ga = 2.0*(hash2(g + vec2(0.0,0.0)) - 0.5);
	vec2 gb = 2.0*(hash2(g + vec2(0.0,1.0)) - 0.5);
	vec2 gc = 2.0*(hash2(g + vec2(1.0,1.0)) - 0.5);
	vec2 gd = 2.0*(hash2(g + vec2(1.0,0.0)) - 0.5);
	
	float da = dot(ga, f - vec2(0.0,0.0));
	float db = dot(gb, f - vec2(0.0,1.0));
	float dc = dot(gc, f - vec2(1.0,1.0));
	float dd = dot(gd, f - vec2(1.0,0.0));
	
	float val = 0.5+0.5*mix(
    	mix(da, dd, inter.x), mix(db, dc, inter.x), inter.y);
  return clamp(val,0.0,1.0);
}

// SDF functions

float sd_sphere(vec3 p, float r) {
  return length(p) - r;
}

// span are the half-lens 
float sd_box(vec3 p, vec3 span) {
  vec3 d = abs(p) - span;
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// positions the box such that the point anchor*span is at the origin
// in this case span are the full-lens
float sd_box(vec3 p, vec3 full_span, vec3 anchor) {
  return sd_box(p + full_span*(anchor - vec3(0.5)), 0.5*full_span);
}

// path circle is on xz plane
float sd_torus(vec3 p, float path_r, float slice_r) {
  vec2 ring_delta = vec2(length(p.xz) - path_r, p.y);
  return length(ring_delta) - slice_r;
}

// aligned with y axis, through the origin
float sd_cylinder(vec3 p, float r) {
  return length(p.xz) - r;
}

// TODO - skip, do not understand the math yet
float sd_cone(vec3 p, vec2 c) {
  return 1000.0;
}

float sd_plane(vec3 p, vec3 plane_pt, vec3 n) {
  // n must be normalized
  return dot(p - plane_pt, n);
}

float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float norm_dist = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - norm_dist * ba) - r;
}

// aligned with the y-axis
float sd_cylinder(vec3 p, vec2 span) {
  vec2 d = vec2(length(p.xz), abs(p.y)) - span;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

/*
float sd_cylinder(vec3 p, vec2 full_span, vec2 anchor) {
  vec2 d = vec2(length(p.xz), abs(p.y)) - span;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
*/

float sd_ellipsoid(vec3 p, vec3 r) {
	float k0 = length(p/r);
	float k1 = length(p/(r*r));
	return k0*(k0-1.0)/k1;
}

vec4 op_elongate(vec3 pos, vec3 extents) {
  vec3 q = abs(pos) - extents;
  return vec4(max(q, 0.0), min(max(max(extents.x, extents.y), extents.z), 0.0));
}

float op_round(float d, float round_amt) {
  return d - round_amt;
}

// assume profile_dist is the distance to the 2d profile
// on the xz plane
float op_extrude(vec3 pos, float profile_dist, float span) {
  vec2 w = vec2(profile_dist, abs(pos.y) - span);
  return length(max(w, 0.0)) + min(max(w.x, w.y), 0.0);
}

// return the position to pass to a 2d sdf when that sdf for that
// sdf to be swept around the y axis
vec2 op_revolution(vec3 pos, float sweep_radius) {
  return vec2(length(pos.xz) - sweep_radius, pos.y);
}

float op_union(float d1, float d2) {
  return min(d1, d2);
}

float op_intersect(float d1, float d2) {
  return max(d1, d2);
}

float op_diff(float d1, float d2) {
  return max(d1, -d2);
}

float op_sunion(float d1, float d2, float k) {
  float h = max(1.0 - abs(d1 - d2) / k, 0.0);
  return min(d1, d2) - h*h*k/4.0;
}

float op_sintersect(float d1, float d2, float k) {
  float h = max(1.0 - abs(d1 - d2) / k, 0.0);
  return max(d1, d2) + h*h*k/4.0;
}

float op_sdiff(float d1, float d2, float k) {
  float h = max(1.0 - abs(d1 + d2) / k, 0.0);
  return max(d1, -d2) + h*h*k/4.0;
}

// operations that tell which one was chosen

vec2 op2_union(float d1, float d2) {
  return vec2(op_union(d1, d2), d1 < d2 ? 0.0 : 1.0);
}

vec2 op2_intersect(float d1, float d2) {
  return vec2(op_intersect(d1, d2), d1 > d2 ? 0.0 : 1.0);
}

vec2 op2_diff(float d1, float d2) {
  return vec2(op_diff(d1, d2), d1 > -d2 ? 0.0 : 1.0);
}

vec2 op2_sunion(float d1, float d2, float k) {
  return vec2(op_sunion(d1,d2,k), d1 < d2 ? 0.0 : 1.0);
}

vec2 op2_sintersect(float d1, float d2, float k) {
  return vec2(op_sintersect(d1,d2,k), d1 > d2 ? 0.0 : 1.0);
}

vec2 op2_sdiff(float d1, float d2, float k) {
  return vec2(op_sdiff(d1,d2,k), d1 > -d2 ? 0.0 : 1.0);
}

// mirror the given position across a plane to the side
// that the plane_nor points at
vec3 mirror(vec3 pos, vec3 plane_pt, vec3 plane_nor) {
  // assume plane_nor is normalized
  float proj = min(dot(pos - plane_pt, plane_nor), 0.0);
  return pos - 2.0 * proj * plane_nor;
}

// for each component, take a if branch is 1 and b if branch is 0
vec3 select(vec3 branch, vec3 a, vec3 b) {
  return branch * a + (1.0 - branch) * b;
}

// linearly repeat the given position
// the first grid cell is centered on the origin, each cell has extent "spacing",
// and the numbers of cells is "count", extending in the positive direction
vec3 repeat(vec3 pos, vec3 origin, vec3 spacing, ivec3 count) {
  vec3 int_part;
  vec3 fract_part = modf((pos - (origin - 0.5*spacing)) / spacing, int_part);
  vec3 int_part_pos = max(sign(int_part), 0.0);
  vec3 cell_pos = spacing * (select(int_part_pos, max(int_part-vec3(count-ivec3(1)),0.0), int_part) + fract_part);
  // reposition the cell_pos such the center of the cell is the origin, not the min corner
  // this allows modeling the repeated geom wrt the origin
  return cell_pos - 0.5 * spacing;
}

vec3 repeat_evenly(vec3 pos, vec3 origin, vec3 span, ivec3 count) {
  // choose a spacing that will evenly space the given number of items along the span
  return repeat(pos, origin, span / max(vec3(count - ivec3(1)), 1.0), count); 
}

// revolve the point around the y axis, placing it in slice
// centered about the x axis
vec3 revolve(vec3 pos, int num_slices) {
  float slice_span = 2.0*pi / float(num_slices);
  float r = length(pos.xz);
  float angle = mod(atan(pos.z, pos.x)+0.5*slice_span, slice_span) - 0.5*slice_span;
  vec2 rotated_pos = r * vec2(cos(angle), sin(angle));
  return vec3(rotated_pos.x, pos.y, rotated_pos.y);
}

// transformations

// convert from world_pos to local_pos, where the local->world transform
// is a rotation about an axis then a translate
vec3 local_pos(vec3 world_pos, vec3 axis, float angle, vec3 trans) {
  // rotation using Rodrigue's rotation formula (see Wikipedia)
  axis = normalize(axis);
  mat3 K = mat3(vec3(0, axis.z, -axis.y), vec3(-axis.z, 0, axis.x), vec3(axis.y, -axis.x, 0.0));
  mat3 rotation_mat = mat3(1.0) + sin(angle)*K + (1.0-cos(angle))*K*K;
  // the local to world transform
  mat4 transform = mat4(rotation_mat);
  transform[3] = vec4(trans, 1.0);
  vec4 local_pos = inverse(transform) * vec4(world_pos, 1.0);
  return local_pos.xyz;
}

float dot2(vec3 v) {
  return dot(v, v);
}

// some helpers

float sd_tube(vec3 pos, vec3 span, vec3 anchor, vec3 offset) {
  // NB: doesn't play well with the anchor is all cases, be careful
  float d = sd_box(pos, span, anchor);
  return op_diff(d, sd_box(pos, span - offset, anchor));
}

vec2 update_res(vec2 cur_res, float d, float obj_id) {
  return (d < cur_res.x) ? vec2(d, obj_id) : cur_res;
}

const float GreyId = 0.0;
const float RedId = 1.0;
const float TestId = 6.0;
const float TerrainId = 7.0;
const float SubsurfaceId = 8.0;

const float MonsterBodyId = 9.0;
const float MonsterMainEyeId = 10.0;
const float MonsterMouthId = 11.0;

// For testing and debugging
vec2 test_sdf(vec3 pos, vec2 res) {
  float d = res.x;
  d = op_union(d, sd_box(pos - vec3(0.0,-1.0,0.0), vec3(4.0,1.0,4.0)));

  // axes
  {
    float len = 20.0;
    float cap_r = 0.05;
    d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(1.0,0.0,0.0), cap_r));
    d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,1.0,0.0), cap_r));
    d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,0.0,1.0), cap_r));
  }
  /*
  // elongation
  {
  vec4 elo = op_elongate(pos, vec3(2.0, 1.0, 1.0));
  return elo.w + sd_sphere(elo.xyz, 1.0);
  }
  {
  vec4 elo = op_elongate(pos, vec3(3.0, 0.0, 3.0));
  return elo.w + sd_torus(elo.xyz, 1.0, 0.2);
  }
  */

  // rounding
  /*
  d = min(d, sd_box(pos, vec3(1.0)));
  d = min(d, op_round(sd_box(pos - vec3(4.0, 0.0, 0.0), vec3(1.0)), 0.5));
  */

  // extrusion
  /*
  float prof_dist = sd_torus(vec3(pos.x, 0.0, pos.z), 2.0, 0.5);
  d = min(d, op_extrude(pos, prof_dist, 3.0));
  */

  // revolution
  /*
  vec2 rev_pos = op_revolution(pos, 4.0);
  d = min(d, sd_box(vec3(rev_pos.x, 0, rev_pos.y), vec3(1.0, 1.0, 1.0)));
  */

  // intersect
  /*
  float d_inter = op_intersect(sd_box(pos, vec3(1.0)), sd_sphere(pos, 1.25)); 
  d = op_union(d, d_inter);
  */

  // diff
  /*
  float d_diff = op_diff(sd_box(pos, vec3(1.0)), sd_sphere(pos, 1.25));
  d = op_union(d, d_diff);
  */

  // smooth operations
  /*
  {
    float d_union = op_sunion(sd_box(pos, vec3(1.0)), sd_sphere(pos - vec3(0.0, 1.0, 0.0), 0.5), 0.5);
    d = op_union(d, d_union);
  }
  */
  /*
  {
    float d_inter = op_sintersect(sd_box(pos, vec3(1.0)), sd_sphere(pos - vec3(0.0, 1.0, 0.0), 0.5), 0.1);
    d = op_union(d, d_inter);
  }
  */
  /*
  {
    float d_diff = op_sdiff(sd_box(pos, vec3(4.0,0.5,4.0)), sd_sphere(pos, 2.0), 0.2);
    d = op_union(d, d_diff);
  }
  */

  // transformations
  /*
  {
    // unif scale
    float scale = 2.0;
    d = op_union(d, sd_sphere(pos / scale, 1.0) * scale);
  }
  */
  /*
  {
    // rotate and translate
    vec3 local = local_pos(pos, vec3(0.0,1.0,0.0), pi/4.0, vec3(5.0));  
    d = op_union(d, sd_box(local, vec3(1.0)));
  }
  */
  
  /*
  // symmetry
  {
    vec3 sym_pos = vec3(abs(pos.x), abs(pos.y), pos.z);
    d = op_union(d, sd_sphere(sym_pos - vec3(3.0), 0.5)); 
  }
  */
  // repetition
  /*
  {
    vec3 rep_pos = vec3(mod(pos.x, 10.0), pos.y, mod(pos.z,4.0)) - vec3(5.0,0.0,2.0);  
    d = op_union(d, sd_sphere(rep_pos, 2.0));
  }
  */
  /*
  // distortion
  {
    float d1 = sd_sphere(pos, 2.0);
    float d2 = 0.1*(sin(10.0*pos.x)+sin(10.0*pos.y)+sin(10.0*pos.z));
    d = op_union(d, d1 + d2);
  }
  */
  /*
  // ambient occlusion
  {
    d = op_union(d, sd_sphere(pos - vec3(0.0,1.0,0.0), 1.0));
  }
  */
  /*
  // mirror
  {
    vec3 m_p = mirror(pos, vec3(0.0), vec3(1.0,0.0,0.0));
    m_p = mirror(m_p, vec3(0.0,0.0,2.0), normalize(vec3(0.0, 0.0, -1.0)));
    // NB: mirror orientation wouldn't matter if also mirrored the center of the
    // sphere, I think. fine for now
    d = op_union(d, sd_sphere(m_p - vec3(2.0,0.0,0.0), 1.0));
  }
  */
  /*
  // linear pattern
  {
    vec3 rep_pos = repeat(pos, vec3(0.0), vec3(1.0,1.0,2.0), ivec3(2,2,2));
    d = op_union(d, sd_sphere(rep_pos, 0.5));
  }
  */
  /*
  // linear pattern even
  {
    vec3 rep_pos = repeat_evenly(pos, vec3(0.0), vec3(4.0,1.0,1.0), ivec3(4,1,1));
    d = op_union(d, sd_sphere(rep_pos, 0.5));
  }
  */
  // revolved pattern
  {
    vec3 rev_pos = revolve(pos, 10);
    d = op_union(d, sd_sphere(rev_pos - vec3(4.0,0.0,0.0), 1.0));
  }
  // anchored box
  /*
  {
    d = op_union(d, sd_box(pos, vec3(4.0,1.0,1.0), vec3(0.5,0.0,1.0)));
  }
  */

  return update_res(res, d, TestId); 
}

vec2 test_shadows(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  d = op_union(d, sd_box(pos, vec3(3.0,5.0,0.5), vec3(0.5,0.0,0.5)));
  d = op_union(d, sd_sphere(pos - vec3(5.0,1.0,0.0), 1.0));

  res = update_res(res, d, GreyId);
  return res;
}

vec2 test_aa(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  res = update_res(res, d, GreyId);
  d = op_union(d, sd_box(pos, vec3(3.0,5.0,0.5), vec3(0.5,0.0,0.5)));
  d = op_union(d, sd_sphere(pos - vec3(5.0,1.0,0.0), 1.0));
  res = update_res(res, d, RedId);

  return res;
}

float terr_fbm(vec2 pos) {
	float freq = 1.0;
	float weight = 0.5;
	float val = 0.0;
	float weight_sum = 0.0;
	for (int i = 0; i < 8; ++i) {
			val += weight * smooth_grad_2d(pos * freq);
			weight_sum += weight;
			weight *= 0.5;
			freq *= 2.0;
	}
	return val / weight_sum;
}

float sd_terrain(vec3 pos) {
  float h = 60.0*terr_fbm(pos.xz/40.0);
  // scale down the SD so that we don't step too far forwards.
  // regardless, we will still stop once we have penetrated the terrain.
  // this doesn't affect the normal since it uniformly scales the
  // iso-surfaces, I think.
  return 0.35*(pos.y - h);  
}

vec2 terrain_sdf(vec3 pos, vec2 res) {
  float d = res.x;
  
  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  res = update_res(res, d, GreyId);

  d = op_union(d, sd_terrain(pos));
  res = update_res(res, d, TerrainId);

  return res;
}

vec2 test_ss(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_sphere(pos, 1.0)); 
  d = op_union(d, sd_box(pos - vec3(6.0,0.0,0.0), vec3(3.0)));
  res = update_res(res, d, SubsurfaceId);

  return res;
}

vec2 debug_sdf(vec3 pos, vec2 res) {
  // axes
  float d = res.x;
  float len = 100.0;
  float cap_r = 0.05;
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(1.0,0.0,0.0), cap_r));
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,1.0,0.0), cap_r));
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,0.0,1.0), cap_r));
  res = update_res(res, d, RedId);
  return res;
}

const float head_h = 18.0;
const float head_r = 10.0;
const vec3 head_pos = vec3(0.0,head_h,0.0);

vec2 monster_sdf(vec3 pos, vec2 in_res) {
  
  // mouth
  vec3 mouth_local = local_pos(pos, vec3(0.0), 0.0, vec3(0.0,head_h+4.0,-head_r)); 
  float m_outer_y = 0.6;
  float m_r_inner = 1.1;
  float mouth_d = sd_ellipsoid(mouth_local, vec3(head_r,m_outer_y*head_r,head_r));
  mouth_d = op_diff(mouth_d, sd_ellipsoid(mouth_local, vec3(m_r_inner*head_r,0.8*m_outer_y*head_r,m_r_inner*head_r)));
  mouth_d = op_intersect(mouth_d, sd_plane(mouth_local, vec3(0.0), vec3(0.0,1.0,0.0)));

  // body
  float body_d = sd_ellipsoid(pos - head_pos, vec3(head_r));
  body_d = op_sunion(body_d, sd_capsule(pos, vec3(0.0), vec3(0.0,head_h,0.0), 4.0), 5.0);

  // legs
  vec3 rev_p = revolve(pos, 6);
  body_d = op_sunion(body_d,
      sd_capsule(rev_p, vec3(0.0), vec3(10.0,0.0,0.0), 2.2), 1.0); 

  // eyes
  vec3 mir_p = mirror(pos, vec3(0.0), vec3(1.0,0.0,0.0));
  float eye_angle = -0.35*pi;
  vec3 eye_pos = 0.7*head_r*vec3(cos(eye_angle),0.0,sin(eye_angle))+vec3(0.0,head_h+3.0,0.0);
  float eye_r = 3.0;
  float eye_d = sd_sphere(mir_p - eye_pos, eye_r);
  float socket_d = sd_sphere(mir_p - eye_pos, eye_r + 0.02);

  // compose
  float d = 1e10;
  vec2 res;
  d = op_union(d, body_d);
  vec2 op_res = op2_sdiff(d, mouth_d, 0.5);
  d = op_res.x;
  d = op_sdiff(d, socket_d, 0.75);
  res = update_res(in_res, d, op_res.y == 0.0 ? MonsterBodyId : MonsterMouthId);
  d = op_union(d, eye_d);
  res = update_res(res, d, MonsterMainEyeId);

  return res;
}

vec2 ground_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  res = update_res(res, d, GreyId);

  return res;
}

vec2 world_sdf(vec3 pos) {
  float d = 1e10;          
  vec2 res = vec2(d, 0.0);

  res = debug_sdf(pos, res);
  //res = ground_sdf(pos, res);
  res = monster_sdf(pos, res);
  //res = test_sdf(pos, res);
  //res = test_aa(pos, res);
  //res = test_ss(pos, res);
  //res = terrain_sdf(pos, res);
  //res = debug_sdf(pos, res);

  return res;
}

float compute_shadow(vec3 ro, vec3 rd, float k) {
  float t = 0.1;
  float res = 1.0;
  for (int i = 0; i < 64; ++i) {
    vec3 pos = ro + t*rd;
    float sd = world_sdf(pos).x;
    res = min(res, k*sd/t);
    t += sd;
    if (res < 0.001 || t > 1000.0) {
      break;
    }
  }
  return clamp(res,0.0,1.0);
}

vec3 world_normal(float obj_id, vec3 pos) {
  vec3 normal = vec3(0.0, 1.0, 0.0);

  // assumes that the sdf is 0.0 at the given pos
  vec2 delta = vec2(0.0, 1.0) * 0.0005;
  normal = normalize(vec3(
    world_sdf(pos + delta.yxx).x -
      world_sdf(pos - delta.yxx).x,
    world_sdf(pos + delta.xyx).x -
      world_sdf(pos - delta.xyx).x,
    world_sdf(pos + delta.xxy).x -
      world_sdf(pos - delta.xxy).x
  ));

  return normal;
}

vec2 world_intersect(vec3 ro, vec3 rd) {
  // TODO - use the boxes to iterate through viable ranges of t

  float t_min = 0.1;
  float t_max = 10000.0;
  float min_step = 0.001;
  // stores (t, obj_id)
  vec2 result = vec2(t_min, -1);
  for (int i = 0; i < 256; ++i) {
    vec3 pt = ro + result.x * rd;
    vec2 dist_result = world_sdf(pt);
    float obj_dist = dist_result.x;
    result.y = dist_result.y;
    // reduce precision of intersection check as distance increases
    if (obj_dist < 0.0001*result.x || result.x > t_max) {
      break;
    }
    result.x += max(obj_dist, min_step);
  }
  if (result.x > t_max) {
    result.y = -1.0;
  }
  return result;
}

float compute_ao(vec3 pos, vec3 nor) {
  float occ = 0.0;
  float weight = 1.0;
  float delta_nor = 0.4 / 5.0;
  for (int i = 0; i < 5; ++i) {
    vec3 sample_pos = pos + nor * float(i) * delta_nor; 
    vec2 dist_result = world_sdf(sample_pos);
    occ += weight * (float(i)*delta_nor - dist_result.x);
    weight *= 0.7;
  }
  return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

vec3 world_color(float obj_id, vec3 ro, vec3 rd, vec3 pos, vec3 normal) {
  vec3 final_color = vec3(-1.0);
  vec3 material_color = vec3(0.5, 0.0, 0.0);
  switch (int(obj_id)) {
    case int(GreyId):
      material_color = vec3(0.5);
      break;
    case int(RedId):
      material_color = vec3(0.5, 0.0, 0.0);
      break;
    case int(TestId): {
      material_color = vec3(0.5,0.0,0.0);
      break;
    }
    case int(MonsterBodyId): {
      material_color = 0.2*vec3(0.43,0.9,0.29);
      break;
    }
    case int(MonsterMainEyeId): {
      vec3 left_orien_vec = vec3(1.0,0.75,-1.5);
      vec3 orien_vec = normalize(
        pos.x > 0.0 ? left_orien_vec : left_orien_vec*vec3(-1.0,1.0,1.0)
      );
      vec3 look_vec = normalize(
        pos.x > 0.0 ? vec3(1.0,1.0,-1.0) : vec3(0.0,0.0,-1.0)
      );
      vec3 pupil_col = vec3(0.83,0.35,0.45);
      vec3 outer_pupil_col = vec3(104.0,58.0,135.0)/255.0;
      // main
      vec3 col = 0.7*vec3(1.0,1.0,0.95);
      // liner
      float dot_orien = clamp(dot(orien_vec, normal),0.0,1.0);
      col = mix(pupil_col, col, smoothstep(0.65,0.7,dot_orien));
      col = mix(outer_pupil_col, col, smoothstep(0.60, 0.65, dot_orien));
      // pupils
      float dot_look = clamp(dot(look_vec, normal),0.0,1.0);
      col = mix(col, vec3(0.0), smoothstep(0.98,1.0,dot_look));
      col = mix(col, pupil_col, smoothstep(0.99,1.0,dot_look));
      
      material_color = col;
      break;
    }
    case int(MonsterMouthId): {
      material_color = 0.2*vec3(0.5,0.01,0.01);
      break;
    }
    case int(TerrainId): {
			vec3 col = vec3(0.05,0.05,0.05);

			vec3 brown = 0.5*vec3(0.45,.30,0.15);
			//vec3 green = 0.63*vec3(0.1,.20,0.10);
      vec3 snow = 0.2*vec3(1.0,0.95,1.0);
			col = mix(col, brown, smoothstep(0.7,1.0,normal.y));
			//col = mix(col, green, smoothstep(0.9,1.0,normal.y));

      float n = smooth_grad_2d(pos.xz/10.0);
      float snow_f = normal.y * pow(pos.y / 10.0,2.0);
      col = mix(col, snow, snow_f*smoothstep(0.5,1.0,n));

      material_color = col;
      break;
    }
    case int(SubsurfaceId): {
      vec3 base_col = vec3(0.3,0.0,0.0);
      vec3 light_dir = vec3(0.0,0.0,1.0);
      vec3 light_col = vec3(1.0,1.0,0.9);
      vec3 scatter_dir = -normalize(light_dir + normal*0.1);
      float light_amt = pow(max(dot(scatter_dir, -rd), 0.0), 100.0);
      // very rough approx of the thinness (AA from inside might work better)
      // works fine if the object is roughly spherical
      float thinness = pow(1.0 - max(dot(scatter_dir, normal),0.0), 2.0);
      light_amt *= thinness;
      base_col = mix(base_col, light_col, light_amt);
      material_color = base_col;

      //final_color = vec3(light_amt);
      break;
    }
  }

  // lighting
  vec3 key_light_col = vec3(1.64,1.27,0.99);
  vec3 key_light_dir = normalize(vec3(1.0, 1.0, 1.0));
  float key_light_amt = clamp(dot(key_light_dir, normal), 0.0, 1.0);
  vec3 fill_light_col = vec3(0.16,0.20,0.28);
  vec3 fill_light_dir = vec3(0.0,1.0,0.0);
  float fill_light_amt = clamp(dot(fill_light_dir, normal), 0.0, 1.0);
  vec3 indir_light_col = 0.25*key_light_col;
  vec3 indir_light_dir = normalize(key_light_dir*vec3(-1.0,0.0,-1.0));
  float indir_light_amt = clamp(dot(indir_light_dir, normal), 0.0, 1.0);

  float occ = compute_ao(pos, normal);
  float shadow = compute_shadow(pos+0.001*key_light_dir, key_light_dir, 16.0);

  vec3 lighting = vec3(0.0);
  lighting += u_bin.x ? vec3(0.0) : key_light_col*key_light_amt*shadow;
  lighting += u_bin.y ? vec3(0.0) : fill_light_col*fill_light_amt*occ;
  lighting += u_bin.z ? vec3(0.0) : indir_light_col*indir_light_amt*occ;

  vec3 out_col = lighting * material_color;

  /*
  // fog
  {
    float pt_dist = length(pos - ro);
    vec3 sun_dir = normalize(vec3(0.0,0.5,1.0));
    float sun_amt = max(dot(key_light_dir, rd), 0.0);
    vec3 fog_color = mix(vec3(0.5,0.6,0.7),
      vec3(1.0,0.9,0.7),pow(sun_amt,2.0));
    float a = 2.0;
    float b = 0.2;
    //float fog_amt = 1.0 - exp(-pt_dist / 500.0); // simple fog
		// TODO - flip the points if nor_y > 0
    float nor_y = min(rd.y, -0.001); // formula requires rd.y < 0
    // elevation-based fog
    //float fog_amt = a*exp(-b*ro.y)*(1.0-exp(-b*nor_y*pt_dist))/(b*nor_y);
    // simple distance fog
    float fog_amt = 1.0-exp(-0.001*pt_dist);
    fog_amt = clamp(fog_amt, 0.0, 1.0);
    out_col = mix(out_col, fog_color, fog_amt);
    //out_col = vec3(fog_amt);
  }
  */

  // gamma correction
  out_col = pow(out_col, vec3(1.0/2.2));

  // if x is not -1.0, we are in some debug mode, so use
  // the debug color
  if (final_color.x == -1.0) {
    final_color = out_col;
  }
  return final_color;
  
  // for debugging:
  //return out_col*(pos.y/10.0);
  //return vec3(1.0) * occ;
  //return vec3(1.0) * shadow;
  //return vec3(1.0) * (normal.x + 1.0) * 0.5;
}

vec3 background_color(vec3 ro, vec3 rd) {
  return vec3(0.29,0.56,0.67);
}

void ray_for_pixel(vec3 eye, vec2 ndc, inout vec3 ro, inout vec3 rd) {
  vec3 look_vec = u_Ref - eye;
  float len = length(look_vec);
  float aspect_ratio = u_Dimensions.x / u_Dimensions.y;
  float v = tan(v_fov) * len;
  float h = aspect_ratio * v; 
  vec3 v_vec = ndc.y * v * u_Up;
  vec3 h_vec = ndc.x * h * cross(look_vec / len, u_Up);

  ro = eye;
  rd = normalize((u_Ref + h_vec + v_vec) - eye);
}

// TODO - change to 2 for a nicer rendering
#define AA 1

void main() {
  vec3 ro, rd;

  vec3 color = vec3(0.0);
  #if AA > 1
  for (int i = 0; i < AA; ++i) {
  for (int j = 0; j < AA; ++j) {
    vec2 offset = 2.0*(vec2(float(i),float(j))/float(AA)-0.5);
    vec2 uv_pos = fs_Pos + offset/u_Dimensions.xy;
  #else
    vec2 uv_pos = fs_Pos;
  #endif
    ray_for_pixel(u_Eye, uv_pos, ro, rd);
    vec2 intersect = world_intersect(ro, rd);
    float t = intersect.x;
    float obj_id = intersect.y;
    if (obj_id == -1.0) {
      color += background_color(ro, rd);  
    } else {
      vec3 inter_pos = ro + t * rd;
      vec3 world_nor = world_normal(obj_id, inter_pos);  
      color += world_color(obj_id, ro, rd, inter_pos, world_nor); 
    } 
  #if AA > 1
  }
  }
  color /= float(AA*AA);
  #endif

  //vec3 col = 0.5 * (rd + vec3(1.0));

  out_Col = vec4(color, 1.0);
}

