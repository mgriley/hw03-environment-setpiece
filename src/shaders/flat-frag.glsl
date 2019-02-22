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

float gradient_noise_3d(vec3 pos) {
	vec3 g = floor(pos);
    vec3 f = fract(pos);
    vec3 i = f*f*f*f*(f*(f*6.0-15.0)+10.0);
    
    vec2 b = vec2(0.0,1.0);
    
    vec3 g000 = 2.0*(hash3(g+b.xxx) - 0.5);
    vec3 g001 = 2.0*(hash3(g+b.xxy) - 0.5);
    vec3 g010 = 2.0*(hash3(g+b.xyx) - 0.5);
    vec3 g011 = 2.0*(hash3(g+b.xyy) - 0.5);
    vec3 g100 = 2.0*(hash3(g+b.yxx) - 0.5);
    vec3 g101 = 2.0*(hash3(g+b.yxy) - 0.5);
    vec3 g110 = 2.0*(hash3(g+b.yyx) - 0.5);
    vec3 g111 = 2.0*(hash3(g+b.yyy) - 0.5);
    
    float d000 = dot(g000,f-b.xxx);
    float d001 = dot(g001,f-b.xxy);
    float d010 = dot(g010,f-b.xyx);
    float d011 = dot(g011,f-b.xyy);
    float d100 = dot(g100,f-b.yxx);
    float d101 = dot(g101,f-b.yxy);
    float d110 = dot(g110,f-b.yyx);
    float d111 = dot(g111,f-b.yyy);
    
    return 0.5+0.5*mix(
    	mix(
        	mix(d000,d100,i.x),
            mix(d010,d110, i.x),
            i.y
        ),
        mix(
        	mix(d001,d101,i.x),
            mix(d011,d111,i.x),
            i.y
        ),
        i.z
    );
}

float voronoi_2d(vec2 pos) {
	vec2 g = floor(pos);
	vec2 f = fract(pos);
	
	float min_dist = 1e10;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			vec2 offset_pos = vec2(ivec2(i,j));
			vec2 delta = offset_pos + hash2(g + offset_pos) - f;
			min_dist = min(min_dist, dot(delta, delta));
		}
	}
	return clamp(sqrt(min_dist),0.0,1.0);
}

float smooth_voronoi_2d(vec2 pos) {
	vec2 g = floor(pos);
	vec2 f = fract(pos);

	float exp_weight = 40.0;
	float sum = 0.0;
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			vec2 offset_pos = vec2(ivec2(i, j));
			vec2 delta = offset_pos + hash2(g + offset_pos) - f;
			sum += exp(-exp_weight*length(delta));            
		}
	}
	return clamp(-(1.0/exp_weight)*log(sum), 0.0, 1.0);
}

float fbm_2d(vec2 pos) {
    float freq = 1.0;
    float weight = 0.5;
    float val = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 3; ++i) {
    	val += weight * gradient_noise_2d(pos * freq);
        weight_sum += weight;
        weight *= 0.5;
        freq *= 2.0;
    }
    return val / weight_sum;
}

float fbm_3d(vec3 pos) {
    float freq = 1.0;
    float weight = 0.5;
    float val = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 3; ++i) {
    	val += weight * gradient_noise_3d(pos * freq);
        weight_sum += weight;
        weight *= 0.5;
        freq *= 2.0;
    }
    return val / weight_sum;
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
// return the new pos in xyz and the slice num of the pt in w
vec4 revolve2(vec3 pos, int num_slices) {
  float slice_span = 2.0*pi / float(num_slices);
  float r = length(pos.xz);
  float cur_angle = atan(pos.z, pos.x)+0.5*slice_span;
  float slice_num = floor(cur_angle / slice_span);
  float angle = mod(cur_angle, slice_span) - 0.5*slice_span;
  vec2 rotated_pos = r * vec2(cos(angle), sin(angle));
  return vec4(rotated_pos.x, pos.y, rotated_pos.y, slice_num);
}

vec3 rotate_y(vec3 pos, float angle) {
  return vec3(cos(angle)*pos.x - sin(angle)*pos.z, pos.y, sin(angle)*pos.x + cos(angle)*pos.z);
}

vec3 revolve(vec3 pos, int num_slices) {
  return revolve2(pos, num_slices).xyz;
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

const float MonsterBodyId = 9.0;
const float MonsterMainEyeId = 10.0;
const float MonsterMouthId = 11.0;

const float LandscapeId = 13.0;
const float PersonId = 15.0;
const float BladeId = 16.0;

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
const float eye_angle = -0.35*pi;
const float eye_r = 3.0;

vec3 monster_eye_pos() {
  return 0.7*head_r*vec3(cos(eye_angle),0.0,sin(eye_angle))+vec3(0.0,head_h+3.0,0.0);
}

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
  int num_legs = 6;
  vec3 rev_p = pos;
  float angle_inc = 2.0*pi / float(num_legs);
  for (int i = 0; i < num_legs; ++i) {
    float leg_num = float(i);
    rev_p = rotate_y(rev_p, angle_inc);
    float cap_r = 2.4;
    float cap_h = 4.0*0.5*(1.0+sin(2.0*2.0*pi*(leg_num + 4.0)/float(num_legs)));
    float cap_z = 0.5*(sin(2.0*2.0*pi*(-leg_num)/float(num_legs))+1.0);
    vec3 pt_b = vec3(10.0, cap_h, cap_z);
    body_d = op_sunion(body_d,
        sd_capsule(rev_p, vec3(0.0), pt_b, cap_r), 2.0); 
    body_d = op_sunion(body_d,
        sd_ellipsoid(rev_p - pt_b, vec3(cap_r)), 2.0);
  }

  // eyes
  vec3 mir_p = mirror(pos, vec3(0.0), vec3(1.0,0.0,0.0));
  vec3 eye_pos = monster_eye_pos();
  float eye_d = sd_sphere(mir_p - eye_pos, eye_r);
  float liner_d = sd_sphere(mir_p - eye_pos, eye_r + 0.001);
  vec3 head_to_eye = normalize(eye_pos - head_pos);
  liner_d = op_sintersect(
    sd_plane(mir_p, eye_pos+0.0*head_to_eye, -head_to_eye), liner_d, 1.0);

  // for clipping to +ve y (allows smoothing the legs as they contact the ground)
  float ground_d = sd_plane(pos, vec3(0.0), vec3(0.0,-1.0,0.0));

  // compose
  float d = 1e10;
  vec2 res;
  d = op_union(d, body_d);
  vec2 op_res = op2_sdiff(d, mouth_d, 0.5);
  d = op_res.x;
  d = op_sintersect(d, ground_d, 2.0);
  res = update_res(in_res, d, op_res.y == 0.0 ? MonsterBodyId : MonsterMouthId);
  d = op_sunion(d, eye_d,0.1);
  res = update_res(res, d, MonsterMainEyeId);

  return res;
}

vec2 person_sdf(vec3 pos, vec2 res) {
  float d = 1e10;

  float scale = 1.0/10.0;
  pos = local_pos(pos, vec3(0.0,1.0,0.0), -pi/10.0, vec3(0.0,0.0,-25.0)) / scale;

  // stick figure in fighting stance
  float torso_len = 5.0;
  float l_r = 0.5; // limb radius
  float head_r = 2.5 * l_r;
  vec3 m_plane = normalize(vec3(-1.0,0.0,-1.0));
  vec3 look_dir = -cross(m_plane, vec3(0.0,1.0,0.0));
  vec3 hip_pos = vec3(0.0,torso_len*1.5,0.0);
  vec3 neck_pos = hip_pos + vec3(0.0,torso_len,0.0) + 1.0*look_dir;
  vec3 head_pos = neck_pos + look_dir*0.75 + vec3(0.0,2.0*head_r,0.0);
  vec3 left_foot_pos = vec3(2.0, 0.0, 2.0);
  vec3 right_foot_pos = left_foot_pos * vec3(-1.0,1.0,-1.0);
  vec3 left_knee_pos = 0.5*(hip_pos + left_foot_pos) + 0.5*look_dir;
  vec3 right_knee_pos = mirror(left_knee_pos, vec3(hip_pos), m_plane);
  vec3 left_elbow_pos = neck_pos + look_dir*2.0 - m_plane*2.0 + vec3(0.0,-3.0,0.0);
  vec3 right_elbow_pos = mirror(left_elbow_pos, vec3(neck_pos), m_plane);
  vec3 left_hand_pos = hip_pos + look_dir*3.0 + vec3(0.0,2.0,2.0);
  vec3 right_hand_pos = left_hand_pos;
  vec3 left_sh_pos = neck_pos - 1.0*m_plane;
  vec3 right_sh_pos = mirror(left_sh_pos, neck_pos, m_plane);

  float s_amt = 0.1;
  d = sd_capsule(pos, hip_pos, neck_pos, l_r);
  d = op_sunion(d, sd_capsule(pos, left_foot_pos, left_knee_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, left_knee_pos, hip_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, right_foot_pos, right_knee_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, right_knee_pos, hip_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, neck_pos, left_sh_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, neck_pos, right_sh_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, left_sh_pos, left_elbow_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, left_elbow_pos, left_hand_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, right_sh_pos, right_elbow_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, right_elbow_pos, right_hand_pos, l_r), s_amt);
  d = op_sunion(d, sd_capsule(pos, neck_pos, head_pos, l_r), s_amt);
  d = op_sunion(d, sd_sphere(pos - head_pos, head_r), 0.125);
  res = update_res(res, d*scale, PersonId);

  // weapon
  vec3 sword_axis = normalize(1.5*look_dir + vec3(-0.5,6.0,0.0));
  vec3 grip_pos = right_hand_pos;
  d = op_union(d, sd_capsule(pos, grip_pos-sword_axis*1.0, grip_pos+sword_axis*0.75, 0.5*l_r));
  res = update_res(res, d*scale, PersonId);
  d = op_union(d, sd_capsule(pos, grip_pos, grip_pos + sword_axis*24.0, 0.2));
  res = update_res(res, d*scale, BladeId);

  return res;
}

float terr_fbm(vec2 pos) {
    float freq = 1.0;
    float weight = 0.5;
    float val = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 1; ++i) {
    	val += weight * smooth_voronoi_2d(pos * freq);
        weight_sum += weight;
        weight *= 0.5;
        freq *= 2.0;
    }
    return val / weight_sum;
}

float mountains_sd(vec3 pos) {
  float h_trig = smoothstep(800.0, 1100.0, length(pos.xz));
  float hot_dist = 1.0 - smoothstep(0.0, 300.0, length(pos.xz - vec2(1000.0,1000.0)));
  float max_h = (1.0+2.0*hot_dist)*h_trig*400.0;
  float h = max_h*terr_fbm(pos.xz/300.0);
  //h = max(h, 0.1*smooth_grad_2d(2.0*pos.xz));
  return 0.25*(pos.y - h);  
}

vec2 ground_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  //d = op_union(d, mountains_sd(pos));
  res = update_res(res, d, LandscapeId);

  return res;
}

vec2 world_sdf(vec3 pos) {
  float d = 1e10;          
  vec2 res = vec2(d, 0.0);

  //res = debug_sdf(pos, res);
  res = ground_sdf(pos, res);
  res = monster_sdf(pos, res);
  res = person_sdf(pos, res);

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

const float T_MAX = 10000.0;

vec2 world_intersect(vec3 ro, vec3 rd) {
  float t_min = 0.1;
  float t_max = T_MAX;
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

const vec3 sun_dir = normalize(vec3(1.0, 0.5, 1.0));

vec3 apply_fog(vec3 in_col, vec3 pos, vec3 ro, vec3 rd) {
  float pt_dist = length(pos - ro);
  float sun_amt = max(dot(sun_dir, rd), 0.0);
  vec3 light_fog = 0.5*vec3(0.88,0.21,0.59);
  vec3 fog_color = mix(0.2*vec3(0.5,0.6,0.7),
    light_fog,pow(sun_amt,2.0));
  float a = 2.0;
  float b = 0.2;

  // distance-fog
  float fog_amt = 1.0 - exp(-pt_dist / 10.0);
  fog_amt *= pow(clamp(abs(1.0-rd.y),0.0,1.0), 6.0);
  //fog_amt *= 1.0 - pow(abs(rd.x), 10.0);

  float n = fbm_3d(rd*6.0 + vec3(0.0,0.0,u_Time/100.0));
  fog_amt *= pow(n,2.0);
  
  fog_amt = clamp(fog_amt, 0.0, 1.0);
  vec3 out_col = mix(in_col, fog_color, fog_amt);
  //out_col = vec3(fog_amt);
  return out_col;
}

// Subsurface Scattering helper - currently unused
float ss_amount(vec3 ro, vec3 rd, vec3 normal, vec3 light_dir) {
  vec3 scatter_dir = -normalize(light_dir + normal*0.1);
  float light_amt = pow(max(dot(scatter_dir, -rd), 0.0), 50.0);
  // very rough approx of the thinness (AA from inside might work better)
  // works fine if the object is roughly spherical
  float thinness = pow(1.0 - max(dot(scatter_dir, normal),0.0), 2.0);
  light_amt *= thinness;
  return light_amt;
}

vec3 world_color(float obj_id, vec3 ro, vec3 rd, vec3 pos, vec3 normal) {
  vec3 final_color = vec3(-1.0);
  vec3 material_color = vec3(0.5, 0.0, 0.0);

  vec3 monster_body_col = 0.2*vec3(0.43,0.9,0.29);
  switch (int(obj_id)) {
    case int(GreyId):
      material_color = vec3(0.5);
      break;
    case int(RedId):
      material_color = vec3(0.5, 0.0, 0.0);
      break;
    case int(MonsterBodyId): {
      material_color = monster_body_col;
      break;
    }
    case int(MonsterMainEyeId): {
      vec3 look_vec = normalize(
        pos.x > 0.0 ? vec3(0.6,0.3,-1.0) : vec3(-1.0,-0.2,-3.2)
      );
      vec3 pupil_col = 0.8*vec3(0.83,0.35,0.45);
      vec3 outer_pupil_col = vec3(104.0,58.0,135.0)/255.0;
      vec3 main_eye_col = 0.7*vec3(1.0,1.0,0.95);

      vec3 left_eye_pos = monster_eye_pos();
      vec3 right_eye_pos = mirror(left_eye_pos, vec3(0.0), vec3(-1.0,0.0,0.0));
      vec3 eye_pos = pos.x > 0.0 ? left_eye_pos : right_eye_pos;
      vec3 eye_out = normalize(pos - eye_pos);
      float dot_orien = clamp(dot(eye_out, normalize(eye_pos - head_pos)),0.0,1.0);

      // progress from body to eye liner to main eye
      vec3 col = monster_body_col;
      col = mix(col, outer_pupil_col, smoothstep(0.7, 0.7 + 0.05, dot_orien));
      col = mix(col, main_eye_col, smoothstep(0.75,0.85, dot_orien));

      // pupils
      float dot_look = clamp(dot(look_vec, eye_out),0.0,1.0);
      col = mix(col, vec3(0.0,0.05,0.0), smoothstep(0.98,0.985,dot_look*dot_look));
      col = mix(col, pupil_col, smoothstep(0.99,0.995,dot_look*dot_look));
      
      material_color = col;
      break;
    }
    case int(MonsterMouthId): {
      material_color = 0.2*vec3(0.5,0.01,0.01);
      break;
    }
    case int(LandscapeId): {
      //vec3 dirt_a = 0.2*vec3(68.0,52.0,27.0)/100.0;
      //vec3 dirt_b = 0.2*vec3(59.0,42.0,17.0)/100.0;

      //vec3 sand = 0.75*vec3(0.83,0.78,0.68);
			vec3 sand = 0.75*vec3(0.4);

      // works well for dirt
      //float n = fbm_2d(2.0*pos.xz);
      //float n2 = fbm_2d(2.0*(pos.xz + vec2(100.0)));

      // attempt at sand
      float n = fbm_2d(8.0*pos.xz);
      float n2 = fbm_2d(8.0*(pos.xz + vec2(100.0)));

      vec3 col = sand;
      //col = mix(col, dirt_b, smoothstep(0.5,0.6,n));
      // arbitrarily perturb the normal of the ground
      // to make the ground look gritty
      float is_ground = smoothstep(0.8,1.0,normal.y);
      normal = normalize(normal + is_ground*0.1*vec3(n-0.5,0.0,n2-0.5));

      material_color = col;
      break;
    }
    case int(PersonId): {
      vec3 col = vec3(0.05);
      material_color = col;
      break;
    }
    case int(BladeId): {
      // make the color > 1.0 so that when gaussian blur is applied
      // there is a very slight bloom effect
      material_color = vec3(6.0);
      break;
    }
  }

  // lighting
  vec3 key_light_col = 2.0*vec3(1.64,1.27,0.99);
  vec3 key_light_dir = sun_dir;
  float key_light_amt = clamp(dot(key_light_dir, normal), 0.0, 1.0);
  vec3 fill_light_col = vec3(0.16,0.20,0.28);
  vec3 fill_light_dir = vec3(0.0,1.0,0.0);
  float fill_light_amt = clamp(dot(fill_light_dir, normal), 0.0, 1.0);
  vec3 indir_light_col = 0.25*key_light_col;
  vec3 indir_light_dir = normalize(key_light_dir*vec3(-1.0,0.0,-1.0));
  float indir_light_amt = clamp(dot(indir_light_dir, normal), 0.0, 1.0);

  float occ_amt = compute_ao(pos, normal);
  vec3 occ = vec3(1.0)*occ_amt;//occ_amt*vec3(104.0,58.0,135.0)/255.0;
  float shadow_amt = compute_shadow(pos+0.001*key_light_dir, key_light_dir, 16.0);
  vec3 shadow = vec3(1.0)*shadow_amt;//shadow_amt*vec3(104.0,58.0,135.0)/255.0;

  vec3 lighting = vec3(0.0);
  lighting += u_bin.x ? vec3(0.0) : key_light_col*key_light_amt*shadow;
  lighting += u_bin.y ? vec3(0.0) : fill_light_col*fill_light_amt*occ;
  lighting += u_bin.z ? vec3(0.0) : indir_light_col*indir_light_amt*occ;

  vec3 out_col = lighting * material_color;

  out_col = apply_fog(out_col, pos, ro, rd);
      
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

float sky_noise(vec2 pos, vec2 focal_pt) {
  vec2 anim_pos = pos + vec2(u_Time / 100.0);
	vec2 g = floor(anim_pos);
  vec2 f = fract(anim_pos);
  
  float min_dist = 1e10;
  for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
          vec2 offset_pos = vec2(ivec2(i,j));
        vec2 delta = offset_pos + hash2(g + offset_pos) - f;
          min_dist = min(min_dist, dot(delta, delta));
      }
  }
  float res = sqrt(min_dist);
  float shape_dist = length(pos - focal_pt) - 3.0;
  float factor = shape_dist > 0.0 ? 8.0 : 12.0;
  res *= exp(-factor*abs(shape_dist));
  return clamp(res,0.0,1.0);
}

vec3 background_color(vec3 ro, vec3 rd) {
  vec3 col = vec3(0.0,0.0,0.05);
  
  // intersect with vertical plane far away, normal (0, 0, -1)
  vec3 plane_pt = ro + rd*(1000.0 - ro.z) / rd.z;

  float n = sky_noise(plane_pt.xy/100.0, vec2(7.0,7.0));
  vec3 light_col = mix(vec3(1.0), vec3(0.88,0.21,0.59), 1.0-rd.y);
  col = vec3(n)*light_col;

  col = apply_fog(col, plane_pt, ro, rd);

  return col;
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

vec4 compute_color(vec2 intersect, vec3 ro, vec3 rd) {
	float t = intersect.x;
  float obj_id = intersect.y;
	vec3 color;
	if (obj_id == -1.0) {
		color = background_color(ro, rd);  
		t = T_MAX;
	} else {
		vec3 inter_pos = ro + t * rd;
		vec3 world_nor = world_normal(obj_id, inter_pos);  
		color = world_color(obj_id, ro, rd, inter_pos, world_nor); 
	}

  // gamma correction
  color = pow(color, vec3(1.0/2.2));

	return vec4(color, t);
}

// Change to 2 to activate AA
#define AA 1

void main() {
  vec3 ro, rd;

  vec3 color = vec3(0.0);
  float t = 0.0;
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
		vec4 res = compute_color(intersect, ro, rd);
    color += res.xyz;
		t += res.w;
  #if AA > 1
  }
  }
  color /= float(AA*AA);
  t /= float(AA*AA);
  #endif

  out_Col = vec4(color, t);
}

