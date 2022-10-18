#version 450

#include "triangle.glsl"

vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0, +1.0),
    vec2(+1.0, -1.0),
    vec2(+1.0, +1.0)
);

vec3 colours[4] = vec3[](
    vec3(0.95, 0.05, 0.05),
    vec3(0.05, 0.95, 0.05),
    vec3(0.05, 0.05, 0.95),
    vec3(0.50, 0.50, 0.50)
);

/*
layout(set = 0, binding = 0)
uniform World {
    vec2 viewport;
    vec2 pos;
    mat4 modelView;
};
*/

const float radius = 0.5;

layout(location = 0)
out vec2 fragVertex;
/*layout(location = 0)
out vec3 fragColour;
layout(location = 1)
out vec2 fragPos[2][2];*/

void main() {
    // vec2 ratio = vec2(min(1.0, viewport.y / viewport.x), min(1.0, viewport.x / viewport.y));
    // vec2 pixelMargin = vec2(1.0, 1.0) / viewport;
    vec2 vertex = positions[gl_VertexIndex];
    // gl_Position = vec4(ratio * (pos + vertex * radius), 0.0, 1.0);
    gl_Position = modelView * vec4(vertex, 0.0, 1.0); // vec4((modelView * vec4(vertex, 0.0, 1.0)).xy, 0.0, 1.0);
    fragVertex = vertex;
    /*fragColour = colours[gl_VertexIndex];
    fragPos[0][0] = vertex + vec2(pixelMargin.x, pixelMargin.y);
    fragPos[0][1] = vertex + vec2(pixelMargin.x, -pixelMargin.y);
    fragPos[1][0] = vertex + vec2(-pixelMargin.x, pixelMargin.y);
    fragPos[1][1] = vertex + vec2(-pixelMargin.x, -pixelMargin.y);*/
}
