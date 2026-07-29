import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import {
  Color,
  Group,
  InstancedMesh,
  MathUtils,
  Object3D,
} from 'three'

import type { ScenePhase } from '../../types/scene'

const PHASE_COLORS: Record<ScenePhase, string> = {
  idle: '#22d3c5',
  retrieving: '#60a5fa',
  generating: '#2de6c9',
  complete: '#6ee7a8',
  insufficient: '#7894a3',
  error: '#9b6670',
  interrupted: '#9b6670',
}

interface FloatingIslandProps {
  phase: ScenePhase
  blockCount: number
}

export function FloatingIsland({
  phase,
  blockCount,
}: FloatingIslandProps) {
  const group = useRef<Group>(null)
  const blocks = useRef<InstancedMesh>(null)
  const targetColor = useMemo(() => new Color(PHASE_COLORS[phase]), [phase])

  const transforms = useMemo(() => {
    const items: Array<[number, number, number, number]> = []
    for (let index = 0; index < blockCount; index += 1) {
      const ring = Math.floor(index / 8)
      const angle = index * 2.39996
      const radius = 0.55 + ring * 0.42 + (index % 3) * 0.12
      items.push([
        Math.cos(angle) * radius,
        -ring * 0.32 - (index % 4) * 0.08,
        Math.sin(angle) * radius * 0.8,
        Math.max(0.48, 0.9 - ring * 0.1),
      ])
    }
    return items
  }, [blockCount])

  useLayoutEffect(() => {
    if (!blocks.current) {
      return
    }
    const object = new Object3D()
    transforms.forEach(([x, y, z, scale], index) => {
      object.position.set(x, y, z)
      object.scale.setScalar(scale)
      object.rotation.y = (index % 4) * (Math.PI / 2)
      object.updateMatrix()
      blocks.current?.setMatrixAt(index, object.matrix)
    })
    blocks.current.instanceMatrix.needsUpdate = true
    blocks.current.computeBoundingSphere()
  }, [transforms])

  useFrame((state, delta) => {
    if (!group.current) {
      return
    }
    const time = state.clock.getElapsedTime()
    group.current.position.y = Math.sin(time * 0.35) * 0.11
    group.current.rotation.y += delta * 0.045
  })

  return (
    <group ref={group} rotation={[0.08, -0.35, -0.04]}>
      <instancedMesh
        ref={blocks}
        args={[undefined, undefined, blockCount]}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[0.68, 0.68, 0.68]} />
        <meshStandardMaterial color="#183e3d" roughness={0.82} metalness={0.08} />
      </instancedMesh>
      <mesh position={[0.1, 0.35, 0]} scale={0.52}>
        <octahedronGeometry args={[0.52, 0]} />
        <meshStandardMaterial
          color={targetColor}
          emissive={targetColor}
          emissiveIntensity={
            phase === 'generating' || phase === 'retrieving' ? 1.15 : 0.62
          }
          roughness={0.28}
        />
      </mesh>
      <pointLight
        position={[0.1, 0.45, 0]}
        color={targetColor}
        intensity={phase === 'error' || phase === 'interrupted' ? 1 : 2.4}
        distance={7}
      />
    </group>
  )
}
