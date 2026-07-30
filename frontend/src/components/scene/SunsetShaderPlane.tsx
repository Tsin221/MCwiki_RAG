import { useFrame, useLoader, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import {
  CanvasTexture,
  ClampToEdgeWrapping,
  Color,
  LinearFilter,
  ShaderMaterial,
  TextureLoader,
  Vector2,
} from 'three'

import type { ScenePhase } from '../../types/scene'

const IMAGE_ASPECT = 16 / 9

const vertexShader = /* glsl */ `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`

const fragmentShader = /* glsl */ `
  varying vec2 vUv;

  uniform sampler2D u_BaseTexture;
  uniform sampler2D u_WaterMask;
  uniform float u_Time;
  uniform float u_WaterSpeed;
  uniform float u_WaterStrength;
  uniform float u_PulseSpeed;
  uniform float u_PulseAmount;
  uniform float u_PulsePower;
  uniform vec3 u_PulseTintLow;
  uniform vec3 u_PulseTintHigh;
  uniform vec2 u_UvScale;
  uniform vec2 u_UvOffset;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 5; i++) {
      value += amplitude * noise(p * frequency);
      frequency *= 2.0;
      amplitude *= 0.6;
    }
    return value;
  }

  void main() {
    vec2 uv = vUv * u_UvScale + u_UvOffset;
    vec4 baseColor = texture2D(u_BaseTexture, uv);
    float waterMask = texture2D(u_WaterMask, uv).r;
    float nearFactor = clamp((0.32 - uv.y) / 0.32, 0.0, 1.0);
    float phase = u_Time * u_WaterSpeed;
    float longWave = sin(uv.y * 82.0 + uv.x * 5.0 + phase * 1.15);
    float crossWave = sin(uv.y * 137.0 - uv.x * 11.0 - phase * 0.72);
    float detailWave = fbm(vec2(
      uv.x * 16.0 + phase * 0.08,
      uv.y * 58.0 - phase * 0.18
    )) * 2.0 - 1.0;

    float ripple = longWave * 0.55 + crossWave * 0.28 + detailWave * 0.17;
    float perspectiveStrength = mix(0.18, 1.0, nearFactor);
    float waterOffset = ripple * u_WaterStrength * perspectiveStrength * waterMask;
    float verticalOffset = (longWave * 0.68 + crossWave * 0.32)
      * u_WaterStrength * 0.32 * perspectiveStrength * waterMask;

    vec2 waterUv = uv + vec2(waterOffset, verticalOffset);
    float destinationMask = texture2D(u_WaterMask, waterUv).r;
    waterUv = mix(uv, waterUv, waterMask * destinationMask);
    vec4 waterColor = texture2D(u_BaseTexture, waterUv);

    float crestSignal = longWave * 0.62 + crossWave * 0.38;
    float waveContrast = crestSignal * mix(0.06, 0.16, nearFactor) * waterMask;
    waterColor.rgb *= 1.0 + waveContrast;

    float crest = pow(clamp(crestSignal * 0.5 + 0.5, 0.0, 1.0), 6.0)
      * waterMask;
    waterColor.rgb += vec3(1.0, 0.48, 0.18)
      * crest * mix(0.05, 0.15, nearFactor);

    float pulseRaw = sin(u_Time * u_PulseSpeed) * 0.5 + 0.5;
    float pulse = smoothstep(0.05, 0.95, pulseRaw) * u_PulseAmount;
    pulse += fbm(vec2(u_Time * 0.041665, u_Time * 0.013885)) * 0.15;
    pulse = pow(clamp(pulse, 0.0, 1.0), u_PulsePower);

    vec3 shimmerColor = mix(u_PulseTintLow, u_PulseTintHigh, pulse);
    vec3 pulsedColor = mix(
      waterColor.rgb,
      waterColor.rgb * shimmerColor,
      pulse * 0.18
    );
    vec3 finalColor = mix(baseColor.rgb, pulsedColor, waterMask);

    gl_FragColor = vec4(finalColor, 1.0);
  }
`

function createWaterMask() {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 256
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Canvas 2D is unavailable')
  }

  const gradient = context.createLinearGradient(0, 0, 0, canvas.height)
  gradient.addColorStop(0, 'rgb(0 0 0)')
  gradient.addColorStop(0.72, 'rgb(0 0 0)')
  gradient.addColorStop(0.75, 'rgb(115 115 115)')
  gradient.addColorStop(0.8, 'rgb(230 230 230)')
  gradient.addColorStop(0.86, 'rgb(255 255 255)')
  gradient.addColorStop(1, 'rgb(255 255 255)')
  context.fillStyle = gradient
  context.fillRect(0, 0, canvas.width, canvas.height)

  const texture = new CanvasTexture(canvas)
  texture.wrapS = ClampToEdgeWrapping
  texture.wrapT = ClampToEdgeWrapping
  texture.minFilter = LinearFilter
  texture.magFilter = LinearFilter
  return texture
}

export function SunsetShaderPlane({ phase }: { phase: ScenePhase }) {
  const material = useRef<ShaderMaterial>(null)
  const imageTexture = useLoader(TextureLoader, '/minecraft-sunset.png')
  const waterMask = useMemo(createWaterMask, [])
  const { width, height } = useThree((state) => state.size)

  const uniforms = useMemo(
    () => ({
      u_BaseTexture: { value: imageTexture },
      u_WaterMask: { value: waterMask },
      u_Time: { value: 0 },
      u_WaterSpeed: { value: 1.05 },
      u_WaterStrength: { value: 0.014 },
      u_PulseSpeed: { value: 1.24 },
      u_PulseAmount: { value: 0.45 },
      u_PulsePower: { value: 0.57 },
      u_PulseTintLow: { value: new Color(1, 0.95, 0.8) },
      u_PulseTintHigh: { value: new Color(1, 0.85, 0.55) },
      u_UvScale: { value: new Vector2(1, 1) },
      u_UvOffset: { value: new Vector2(0, 0) },
    }),
    [imageTexture, waterMask],
  )

  useEffect(() => {
    imageTexture.minFilter = LinearFilter
    imageTexture.magFilter = LinearFilter
    imageTexture.needsUpdate = true
  }, [imageTexture])

  useEffect(() => {
    const viewportAspect = width / height
    const scale = uniforms.u_UvScale.value
    const offset = uniforms.u_UvOffset.value

    if (viewportAspect > IMAGE_ASPECT) {
      scale.set(1, IMAGE_ASPECT / viewportAspect)
    } else {
      scale.set(viewportAspect / IMAGE_ASPECT, 1)
    }
    offset.set((1 - scale.x) / 2, (1 - scale.y) / 2)
  }, [height, uniforms, width])

  useEffect(() => () => waterMask.dispose(), [waterMask])

  useFrame((state) => {
    if (!material.current) {
      return
    }
    material.current.uniforms.u_Time.value = state.clock.getElapsedTime()
    material.current.uniforms.u_PulseAmount.value =
      phase === 'retrieving' || phase === 'generating' ? 0.58 : 0.45
  })

  return (
    <mesh frustumCulled={false}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        depthTest={false}
        depthWrite={false}
      />
    </mesh>
  )
}
