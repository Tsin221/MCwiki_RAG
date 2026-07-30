import { Canvas } from '@react-three/fiber'

import { usePageVisible } from '../../hooks/usePageVisible'
import { useSceneQuality } from '../../hooks/useSceneQuality'
import type { ScenePhase } from '../../types/scene'
import { SunsetShaderPlane } from './SunsetShaderPlane'

export default function AmbientScene({ phase }: { phase: ScenePhase }) {
  const quality = useSceneQuality()
  const pageVisible = usePageVisible()

  return (
    <div className="scene-canvas" aria-hidden="true">
      <Canvas
        dpr={quality.dpr}
        frameloop={pageVisible ? 'always' : 'never'}
        orthographic
        camera={{ position: [0, 0, 1], near: 0.1, far: 10 }}
        gl={{
          alpha: true,
          antialias: false,
          powerPreference: 'high-performance',
        }}
      >
        <SunsetShaderPlane phase={phase} />
      </Canvas>
    </div>
  )
}
