import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import { Points } from 'three'

function pseudoRandom(index: number) {
  const value = Math.sin(index * 91.732 + 17.13) * 43758.5453
  return value - Math.floor(value)
}

export function SceneParticles({ count }: { count: number }) {
  const points = useRef<Points>(null)
  const positions = useMemo(() => {
    const values = new Float32Array(count * 3)
    for (let index = 0; index < count; index += 1) {
      values[index * 3] = (pseudoRandom(index) - 0.5) * 9
      values[index * 3 + 1] = (pseudoRandom(index + 100) - 0.5) * 6
      values[index * 3 + 2] = (pseudoRandom(index + 200) - 0.5) * 5
    }
    return values
  }, [count])

  useFrame((_, delta) => {
    if (points.current) {
      points.current.rotation.y += delta * 0.012
    }
  })

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color="#6ee7d8"
        size={0.035}
        transparent
        opacity={0.62}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  )
}
