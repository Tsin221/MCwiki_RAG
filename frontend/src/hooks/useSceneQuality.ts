import { useEffect, useState } from 'react'

import type { SceneQuality } from '../types/scene'

interface DeviceMetrics {
  viewportWidth: number
  hardwareConcurrency: number
}

export function chooseSceneQuality({
  viewportWidth,
  hardwareConcurrency,
}: DeviceMetrics): SceneQuality {
  const isLowQuality = viewportWidth <= 820 || hardwareConcurrency <= 4
  return isLowQuality
    ? {
        level: 'low',
        dpr: [1, 1],
      }
    : {
        level: 'high',
        dpr: [1, 1.5],
      }
}

function readMetrics(): DeviceMetrics {
  return {
    viewportWidth: window.innerWidth,
    hardwareConcurrency: navigator.hardwareConcurrency || 4,
  }
}

export function useSceneQuality(): SceneQuality {
  const [quality, setQuality] = useState(() =>
    chooseSceneQuality(readMetrics()),
  )

  useEffect(() => {
    const update = () => setQuality(chooseSceneQuality(readMetrics()))
    window.addEventListener('resize', update, { passive: true })
    return () => window.removeEventListener('resize', update)
  }, [])

  return quality
}
