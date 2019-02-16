#version 300 es
precision highp float;

uniform bvec3 u_bin;
uniform vec3 u_fin;
uniform sampler2D u_tex;

in vec2 fs_Pos;
out vec4 out_Col;

void main() {
  vec2 uv = (fs_Pos + 1.0)*0.5;
  vec4 coord = texture(u_tex, uv);
  vec3 col = coord.rgb;
  float dist = coord.a;
  
  col = vec3(dist / 100.0);

  out_Col = vec4(col, 1.0);
}

