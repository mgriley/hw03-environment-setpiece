import {vec2, vec3} from 'gl-matrix';
import * as Stats from 'stats-js';
import * as DAT from 'dat-gui';
import Square from './geometry/Square';
import OpenGLRenderer from './rendering/gl/OpenGLRenderer';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  bool_a: false,
  bool_b: false,
  bool_c: false,
  float_a: 0.0,
  float_b: 0.0,
  float_c: 0.0
};

let square: Square;
let time: number = 0;

// some convenient camera controls for switching to/from the
// view that the final scene will be seen from
let cam_render_state_old: any = {
  eye: vec3.fromValues(-1.8, 6.5, -32.0),
  target: vec3.fromValues(-1.6, 3.6, 0.4)
};
let cam_render_state: any = {
  eye: vec3.fromValues(-6.7, 1.6, -31.70),
  target: vec3.fromValues(-1.6, 3.6, 0.4)
};
let cam_mid_state: any = {
  eye: vec3.fromValues(10, 30, -50),
  target: vec3.fromValues(0, 0, 0)
};
let cam_far_state: any = {
  eye: vec3.fromValues(130, 288, -369),
  target: vec3.fromValues(0, 0, 0)
};
let cam_prev_state: any = {
  eye: vec3.fromValues(10, 30, -50),
  target: vec3.fromValues(0, 0, 0)
};
let cam_to_render_pos: boolean = false;
let cam_to_last_pos: boolean = false;
let cam_to_mid_pos: boolean = false;
let cam_to_far_pos: boolean = false;

function set_camera_from_state(cam: any, state: any) {
  cam.controls.lookAt(state.eye, state.target, [0, 1, 0]);
}

function loadScene() {
  square = new Square(vec3.fromValues(0, 0, 0));
  square.create();
  // time = 0;
}

function main() {
  window.addEventListener('keypress', function (e) {
    //console.log('pressed: ', e.key);
    switch(e.key) {
      case 'r':
        cam_to_render_pos = true;
        break;
      case 't':
        cam_to_last_pos = true;
        break;
      case 'f':
        cam_to_mid_pos = true;
        break;
      case 'g':
        cam_to_far_pos = true;
        break;
      case 'c':
        let eye = camera.controls.eye;
        let center = camera.controls.center;
        let up = camera.controls.up;
        console.log(`eye: ${eye}\ntarget: ${center}\nup: ${up}`);
        break;
    }
  }, false);

  window.addEventListener('keyup', function (e) {
    switch(e.key) {
      // Use this if you wish
    }
  }, false);

  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // Add controls to the gui
  const gui = new DAT.GUI();
  gui.add(controls, 'bool_a');
  gui.add(controls, 'bool_b');
  gui.add(controls, 'bool_c');
  gui.add(controls, 'float_a', 0, 1);
  gui.add(controls, 'float_b', 0, 1);
  gui.add(controls, 'float_c', 0, 1);

  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');
  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }
  // this is required to render to floating point format textures
  var ext = gl.getExtension('EXT_color_buffer_float');
  if (!ext) {
    alert('EXT_color_buffer_float not supported!');
  }

  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  // Initial call to load scene
  loadScene();

  const camera = new Camera(vec3.fromValues(10.0, 30, -50), vec3.fromValues(0, 0, 0));
  set_camera_from_state(camera, cam_render_state);

  const renderer = new OpenGLRenderer(canvas, window.innerWidth, window.innerHeight);
  gl.enable(gl.DEPTH_TEST);

  const flat = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/flat-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/flat-frag.glsl')),
  ]);
  const post = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/post-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/post-frag.glsl')),
  ]);

  function processKeyPresses() {
    if (cam_to_render_pos) {
      cam_to_render_pos = false;
      vec3.copy(cam_prev_state.eye, camera.controls.eye);
      vec3.copy(cam_prev_state.target, camera.controls.center);
      set_camera_from_state(camera, cam_render_state);
    }
    if (cam_to_last_pos) {
      cam_to_last_pos = false;
      set_camera_from_state(camera, cam_prev_state);
    }
    if (cam_to_mid_pos) {
      cam_to_mid_pos = false;
      set_camera_from_state(camera, cam_mid_state);
    }
    if (cam_to_far_pos) {
      cam_to_far_pos = false;
      set_camera_from_state(camera, cam_far_state);
    }
  }

  // This function will be called every frame
  function tick() {
    camera.update();
    stats.begin();
    processKeyPresses();
    flat.setControls(controls);
    post.setControls(controls);
    renderer.render(camera, flat, post, square, time);
    time++;
    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
    flat.setDimensions(window.innerWidth, window.innerHeight);
  }, false);

  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();
  flat.setDimensions(window.innerWidth, window.innerHeight);

  // Start the render loop
  tick();
}

main();
