#version 300 es
precision highp float;

uniform bvec3 u_bin;
uniform vec3 u_fin;
uniform sampler2D u_tex;

in vec2 fs_Pos;
out vec4 out_Col;

const float pi = 3.14159;

vec3 apply_blur(vec2 uv, vec3 in_col, float in_dist) {
  // NB: this implementation is unoptimized.
  // see: http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/

  float focal_dist = 30.0;
  float blur_amt = min(abs(in_dist - focal_dist) / 10.0, 1.0);
  float std_dev = 10.0*blur_amt;
  vec3 col_sum = vec3(0.0);
  float weight_sum = 0.0;
  int kernel_size = 3;
  for (int i = -kernel_size; i <= kernel_size; ++i) {
    for (int j = -kernel_size; j <= kernel_size; ++j) {
      vec2 p = vec2(float(i), float(j));
      vec2 sample_uv = uv + 0.0005*p;
      vec4 res = texture(u_tex, sample_uv);
      float w = exp(-(p.x*p.x+p.y*p.y)/(2.0*std_dev*std_dev)) / (2.0*pi*std_dev*std_dev);
      col_sum += w*res.xyz;
      weight_sum += w;
    }
  }
  col_sum /= weight_sum;
  return col_sum;
}

void main() {
  vec2 uv = (fs_Pos + 1.0)*0.5;
  vec4 coord = texture(u_tex, uv);
  vec3 in_col = coord.rgb;
  float in_dist = coord.a;
  
  vec3 blur_col = apply_blur(uv, in_col, in_dist);
  vec3 col = u_bin.x ? in_col : blur_col;
  //vec3 col = in_col;

  out_Col = vec4(col, 1.0);
}

