import {mat4, vec4} from 'gl-matrix';
import Drawable from './Drawable';
import Camera from '../../Camera';
import {gl} from '../../globals';
import ShaderProgram from './ShaderProgram';

// In this file, `gl` is accessible because it is imported above
class OpenGLRenderer {
  fbo: WebGLFramebuffer;
  render_texture: WebGLTexture;
  depth_buffer: WebGLRenderbuffer;

  constructor(public canvas: HTMLCanvasElement, w: number, h: number) {
    // setup buffers for two-stage rendering
    
    this.render_texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.render_texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    this.depth_buffer = gl.createRenderbuffer()
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depth_buffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);

    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depth_buffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.render_texture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) {
      console.error('framebuffer is  not complete')
    }
  }

  setClearColor(r: number, g: number, b: number, a: number) {
    gl.clearColor(r, g, b, a);
  }

  setSize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    // resize fbo attachments
    gl.bindTexture(gl.TEXTURE_2D, this.render_texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.depth_buffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  }

  render(camera: Camera, prog: ShaderProgram, post_prog: ShaderProgram, square: Drawable, time: number) {
    // render to texture
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    prog.setEyeRefUp(camera.controls.eye, camera.controls.center, camera.controls.up);
    prog.setTime(time);

    prog.draw(square)

    // post-processing
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // texture unit 0 is active by default
    prog.setTextureUnit(0);
    gl.bindTexture(gl.TEXTURE_2D, this.render_texture);
    post_prog.draw(square);
  }
};

export default OpenGLRenderer;
