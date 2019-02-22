# CIS 566 Homework 3: Environment Setpiece

## Overview

Matthew Riley\
PennKey: matriley\
Live at: https://mgriley.github.io/hw03-environment-setpiece/

Please note: The scene renders as intended on my Mac laptop and a Mac desktop, but not on the one windows desktop that I tried (in which case the monster is mouth-less). I'm unsure what causes this.

![](demo_shot.png)

## Description:

## Techniques:

* Animation: the light at the rim of the eclipse and the the fog density are slowly animated
* Noise:
  - The flares of the eclipse are made using voronoi noise (in an attempt to simulate lens flare)
  - The graininess of the ground is made by randomly perturbing the surface normal
  - The fog density is is a function of the ray direction
* Remapping of [0, 1] to a set of colors: not done
* Toolbox functions: smoothstep is used frequently
* Lighting: three light sources are used
* Penumbra shadows: as seen on the monster and warrior
* Ambient Occulsion: using the 5-tap method. Can be seen under the monster's feet.
* SDF Blending: the monster is modeled using blended SDFs (see the monster_sdf in flat-frag.glsl)
* Anti-aliasing: the screenshot was taking of the scene rendered with AA on, but it is left off by default to allow the scene to render at a reasonable frame rate
* Post-processing: a depth-of-field effect is applied, where the monster's head is near the focal point. The rendering passes are implemented in OpenGLRenderer.ts.

## Sources

https://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf
https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
https://www.iquilezles.org/www/articles/smin/smin.htm
https://www.iquilezles.org/www/articles/raymarchingdf/raymarchingdf.htm
https://www.iquilezles.org/www/articles/sdfmodeling/sdfmodeling.htm
http://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
https://www.iquilezles.org/www/articles/functions/functions.htm
http://iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm
http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
http://iquilezles.org/www/articles/fog/fog.htm
https://www.shadertoy.com/view/4tByz3
https://www.shadertoy.com/view/Xds3zN

