export type ScenePhase =
  | 'idle'
  | 'retrieving'
  | 'generating'
  | 'complete'
  | 'insufficient'
  | 'error'
  | 'interrupted'

export interface SceneQuality {
  level: 'high' | 'low'
  dpr: [number, number]
  blockCount: number
  particleCount: number
}
