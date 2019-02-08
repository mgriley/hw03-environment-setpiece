#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

const float pi = 3.14159;
const float v_fov = pi / 4.0;

struct AABB {
  vec3 min;
  vec3 max;
};

// for testing:
const int num_objects = 2;
const vec3 sphere_center = vec3(0.0);
const vec3 sphere_center_b = vec3(10.0);
const vec3 box_center = vec3(-4.0, 0.0, 10.0);
const vec3 torus_center = vec3(4.0, 0.0, 10.0);

// Noise functions

float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float random1( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

vec2 random2( vec2 p , vec2 seed) {
  return fract(sin(vec2(dot(p + seed, vec2(311.7, 127.1)), dot(p + seed, vec2(269.5, 183.3)))) * 85734.3545);
}

float surflet_noise_3d(vec3 in_pt, vec2 seed) {
  vec2 p = in_pt.xy;
  p += vec2(in_pt.z, 0.0);

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
    vec2 corner = scale * corners[i];

    vec2 corner_dir = 2.0 * random2(corner, seed) - vec2(1.0);
    // rotate the vector by the third component
    float angle = in_pt.z;
    mat2 rotate_matrix = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    corner_dir = rotate_matrix * corner_dir;

    vec2 delta = p - corner;
    // this is the height if we were only on a slope of
    // magnitude length(corner_dir) in the direction of corner_dir
    float sloped_height = dot(delta, corner_dir);
    float weight = 1.0 - smoothstep(0.0, scale, length(delta));
    sum += 0.25 * weight * sloped_height;
  }
  return (sum + 1.0) / 2.0;
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

// Engine functions

vec2 update_res(vec2 cur_res, float d, float obj_id) {
  return (d < cur_res.x) ? vec2(d, obj_id) : cur_res;
}

// For testing and debugging
vec2 world_sdf_test(vec3 pos) {
  float d = 1e10;
  d = op_union(d, sd_box(pos - vec3(0.0,-1.0,0.0), vec3(4.0,1.0,4.0)));

  vec2 res = vec2(d, 1.0);

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
  /*
  // revolved pattern
  {
    vec3 rev_pos = revolve(pos, 10);
    d = op_union(d, sd_sphere(rev_pos - vec3(4.0,0.0,0.0), 1.0));
  }
  */
  // anchored box
  /*
  {
    d = op_union(d, sd_box(pos, vec3(4.0,1.0,1.0), vec3(0.5,0.0,1.0)));
  }
  */

  if (d < res.x) {
    res.x = d;
    res.y = 0.0;
  }

  return res;
}

const float GreyId = 0.0;
const float RedId = 1.0;
const float CastleId = 2.0;
const float BridgeId = 3.0;
const float DoorId = 4.0;
const float GroundId = 5.0;

float turret_sdf(vec3 pos) {
  float d = 1e10;
  float slen = 4.0;
  float tur_h = 15.0;
  vec3 tur_span = vec3(slen,tur_h,slen);
  float wall_th = 0.25;
  vec3 offset = vec3(wall_th,-wall_th,wall_th);
  d = op_union(d, sd_tube(pos, tur_span, vec3(0.5,0.0,0.5), offset));

  // windows
  vec3 rev_pos = revolve(pos, 4);
  vec3 m_pos = mirror(rev_pos, vec3(0.0), vec3(0.0,0.0,1.0));
  vec3 win_pos = vec3(0.0, 12.5, 1.0);
  d = op_sdiff(d, sd_box(m_pos - win_pos, vec3(slen+1.0,2.0,0.5), vec3(0.0,0.5,0.5)), 0.1);

  // ridges
  vec3 rid_pos = mirror(pos, vec3(0.0), normalize(vec3(1.0,0.0,-1.0)));
  rid_pos = mirror(rid_pos, vec3(0.0), normalize(vec3(-1.0,0.0,-1.0)));
  int num_ridges = 4;
  rid_pos = repeat(rid_pos, vec3(-slen*0.5,tur_h,-slen*0.5), vec3(slen/float(num_ridges),1.0,1.0),ivec3(num_ridges,1,1));
  d = op_union(d, sd_box(rid_pos, vec3(wall_th*1.5,wall_th*2.0,0.5*wall_th), vec3(0.0)));

  return d;
}

vec2 castle_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  float main_len = 30.0;
  float main_h = 10.0;
  vec3 main_offset = vec3(1.0,-0.25,1.0);
  d = op_union(d, sd_tube(pos, vec3(main_len,main_h,main_len), vec3(0.5,0.0,0.5), main_offset));
  
  vec3 m_pos = mirror(pos, vec3(0.0), vec3(0.0,0.0,1.0));
  m_pos = mirror(m_pos, vec3(0.0), vec3(1.0,0.0,0.0));
  vec3 turret_offset = vec3(main_len*0.5, 0.0, main_len*0.5);
  d = op_union(d, turret_sdf(m_pos - turret_offset));
  
  //d = op_sunion(res.x, d, 0.5);
  return update_res(res, d, CastleId);
}

vec2 bridge_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  // draw bridge
  float bridge_angle = 0.5*(sin(u_Time/10.0) + 1.0) * pi/2.0;
  vec3 lpos = local_pos(pos, vec3(0.0,0.0,1.0), bridge_angle, vec3(0.5*30.0,0.0,0.0));
  d = op_union(d, sd_box(lpos, vec3(10.0,0.2,4.0), vec3(0.0,0.0,0.5)));

  return update_res(res, d, BridgeId);
}

vec2 door_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_box(pos - vec3(0.5*30.0,0.0,0.0), vec3(0.05,6.0,4.0), vec3(0.0,0.0,0.5)));

  return update_res(res, d, DoorId);
}

vec2 ground_sdf(vec3 pos, vec2 res) {
  float d = res.x;

  d = op_union(d, sd_plane(pos, vec3(0.0), vec3(0.0,1.0,0.0)));
  float moat_ext = 20.0;
  vec4 elo = op_elongate(pos, vec3(moat_ext,2.0,moat_ext));
  float moat_d = elo.w + sd_torus(elo.xyz, 2.0, 1.0);
  d = op_sdiff(d, moat_d, 1.0);

  res = update_res(res, d, GroundId);
  return res;
}

vec2 debug_sdf(vec3 pos, vec2 res) {
  // axes
  float d = res.x;
  float len = 20.0;
  float cap_r = 0.05;
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(1.0,0.0,0.0), cap_r));
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,1.0,0.0), cap_r));
  d = op_union(d, sd_capsule(pos, vec3(0.0), len*vec3(0.0,0.0,1.0), cap_r));
  res = update_res(res, d, RedId);
  return res;
}

vec2 world_sdf(vec3 pos) {
  float d = 1e10;          
  vec2 res = vec2(d, 0.0);

  //res = debug_sdf(pos, res);
  res = ground_sdf(pos, res);
  res = castle_sdf(pos, res);
  res = bridge_sdf(pos, res);
  res = door_sdf(pos, res);

  return res;
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

  // BB should be a list of (t_min, t_max) pairs for each box
  /*
  // TODO - safe to assume that they are default-initialized?
  AABB[num_objects] boxes;
  // TODO - set the boxes for the objects that have them,
  // and leave the rest uninitialized

  // use the BB to compute the list of active object ids
  int num_active = 0;
  float[num_objects] active_objects;
  for (int i = 0; i < boxes.length(); ++i) {
    AABB box = boxes[i];
    if (box.max - box.min != vec3(0.0) || intersects_bb(ro, rd, box)) {
      active_objects[num_active] = i;  
    }
  }

  // activate all objects
  float[num_objects] active_objects;
  for (int i = 0; i < num_objects; ++i) {
    active_objects[i] = float(i);
  }
  int num_active = num_objects;
  */

  float t_min = 0.1;
  float t_max = 1000.0;
  int max_steps = 200;
  float min_step = 0.001;
  // stores (t, obj_id)
  vec2 result = vec2(t_min, -1);
  for (int i = 0; i < max_steps; ++i) {
    vec3 pt = ro + result.x * rd;
    vec2 dist_result = world_sdf(pt);
    float obj_dist = dist_result.x;
    result.y = dist_result.y;
    // reduce precision of intersection check as distance increases
    if (abs(obj_dist) < 0.0001*result.x || result.x > t_max) {
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

vec3 world_color(float obj_id, vec3 pos, vec3 normal) {
  vec3 material_color = vec3(0.5, 0.0, 0.0);
  switch (int(obj_id)) {
    case int(GreyId):
      material_color = vec3(0.5);
      break;
    case int(RedId):
      material_color = vec3(0.5, 0.0, 0.0);
      break;
    case int(CastleId):
      material_color = vec3(0.6);
      break;
    case int(BridgeId): {
      float factor = 1.0-pow(smoothstep(0.0,1.0,sin(10.0*pos.z)), 50.0);
      material_color = factor*vec3(0.4,0.3,0.2);
      break;
    }
    case int(DoorId):
      material_color = vec3(0.4,0.3,0.2);
      break;
    case int(GroundId): {
      float noise = surflet_noise(5.0*pos.xz, vec2(23.1, 781.0));
      vec3 grass_color = mix(vec3(0.13,0.39,0.11), vec3(0.07,0.24,0.06), noise);
      vec3 slope_color = mix(vec3(0.48,0.37,0.14), vec3(0.30,0.22,0.08), noise);
      vec3 col = mix(slope_color, grass_color, smoothstep(0.0,1.0,normal.y));
      material_color = col;
      break;
    }
  }

  vec3 light_dir = normalize(vec3(-1.0, 1.0, -1.0));
  float diffuse_factor = clamp(dot(light_dir, normal), 0.0, 1.0);
  float ambient_factor = 0.2;
  float occ = compute_ao(pos, normal);

  return material_color * (diffuse_factor + ambient_factor) * occ;
  
  // for debugging:
  //return vec3(1.0) * occ;
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

vec3 compute_eye() {
  return u_Eye;
  // this looks awful
  // animate the camera around the scene
  /*
  float angle = u_Time/200.0;
  vec2 eye_xz = mat2(vec2(cos(angle), sin(angle)), vec2(-sin(angle), cos(angle))) * u_Eye.xz;
  return vec3(eye_xz.x, u_Eye.y, eye_xz.y);
  */
}

void main() {
  vec3 ro, rd;

  vec3 eye = compute_eye();  
  ray_for_pixel(eye, fs_Pos, ro, rd);

  vec2 intersect = world_intersect(ro, rd);
  float t = intersect.x;
  float obj_id = intersect.y;
  vec3 color;
  if (obj_id == -1.0) {
    color = background_color(ro, rd);  
  } else {
    vec3 inter_pos = ro + t * rd;
    vec3 world_nor = world_normal(obj_id, inter_pos);  
    color = world_color(obj_id, inter_pos, world_nor); 
  }

  //vec3 col = 0.5 * (rd + vec3(1.0));

  out_Col = vec4(color, 1.0);
}

