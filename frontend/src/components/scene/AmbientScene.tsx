import { Canvas } from '@react-three/fiber'

import { usePageVisible } from '../../hooks/usePageVisible'
import { useSceneQuality } from '../../hooks/useSceneQuality'
import type { ScenePhase } from '../../types/scene'
import { FloatingIsland } from './FloatingIsland'
import { SceneParticles } from './SceneParticles'

export default function AmbientScene({ phase }: { phase: ScenePhase }) {
  const quality = useSceneQuality()
  const pageVisible = usePageVisible()

  return (
    <div className="scene-canvas" aria-hidden="true">
      <Canvas
        dpr={quality.dpr}
        frameloop={pageVisible ? 'always' : 'never'}
        camera={{ position: [0, 1.1, 5.6], fov: 42, near: 0.1, far: 30 }}
        gl={{
          alpha: true,
          antialias: quality.level === 'high',
          powerPreference: 'high-performance',
        }}
      >
        <fog attach="fog" args={['#020711', 5, 13]} />
        <ambientLight intensity={0.55} color="#8fb5c8" />
        <directionalLight
          position={[3, 5, 4]}
          intensity={2.1}
          color="#b8f4e9"
        />
        <directionalLight
          position={[-4, 0, -3]}
          intensity={0.8}
          color="#447dcc"
        />
        <group position={[2.05, -0.2, 0]}>
          <FloatingIsland phase={phase} blockCount={quality.blockCount} />
        </group>
        <SceneParticles count={quality.particleCount} />
      </Canvas>
    </div>
  )
}
