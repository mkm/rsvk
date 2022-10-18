#version 450

#include "triangle.glsl"
#include "sobol.glsl"

layout(location = 0)
in vec2 inVertex;
/*layout(location = 0)
in vec3 inColour;
layout(location = 1)
in vec2 inPos[2][2];*/

layout(location = 0)
out vec4 outColour;

vec4 eval(vec2 vertex) {
    float distSqr = dot(vertex, vertex);
    if (distSqr <= 1) {
        float param = mod(atan(vertex.y, vertex.x) / PI, 0.25);
        if (param <= 0.025 || param >= 0.225 || distSqr >= 0.85 || distSqr <= 0.01) {
            return vec4(0.75, 0.5, 0.75, 1.0);
        } else {
            return vec4(0.75, 0.25, 0.25, 1.0);
        }
    } else {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
}

void main() {
    outColour = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; ++i) {
        vec2 screenPos = 2.0 * (gl_FragCoord.xy - vec2(0.5, 0.5) + sobolNumbers[i]) / viewport - 1.0;
        vec2 vertex = (inverse(modelView) * vec4(screenPos, 0.0, 1.0)).xy;
        outColour += eval(vertex) / numSamples;
    }
    /*if (dot(vertex, vertex) <= 1) {
        outColour = vec4(0.75, 0.25, 0.25, 1.0);
    } else {
        outColour = vec4(0.0, 0.0, 0.0, 0.0);
    }*/
    // outColour = vec4(vertex * 0.125, 0.0, 1.0);
    /*for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            vec2 pos = inPos[r][c];
            if (dot(pos, pos) <= 1.0) {
                outColour += 0.25 * vec4(0.75, 0.25, 0.25, 1.0);
                // outColour += 0.25 * vec4(1.0, 0.0, -1.0, 1.0);
            }
        }
    }
    outColour = vec4(1.0, 0.0, 0.0, 1.0);*/
    /*if (dot(inPos, inPos) > 1.0) {
        discard;
    }
    float scale = dot(inPos, inPos); // 2.0 * max(0.0, length(inPos) - 0.5);
    outColour = mix(vec4(0.75, 0.65, 0.75, 1.0), vec4(0.75, 0.25, 0.25, 1.0), scale);*/
}
