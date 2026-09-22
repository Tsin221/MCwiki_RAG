import { describe, expect, it } from 'vitest'

import { chooseSceneQuality } from './useSceneQuality'

describe('chooseSceneQuality', () => {
  it('uses a reduced mobile scene on narrow or low-core devices', () => {
    expect(
      chooseSceneQuality({ viewportWidth: 360, hardwareConcurrency: 8 }),
    ).toEqual({
      level: 'low',
      dpr: [1, 1],
    })
    expect(
      chooseSceneQuality({ viewportWidth: 1440, hardwareConcurrency: 4 }),
    ).toMatchObject({ level: 'low' })
  })

  it('caps desktop pixel ratio at 1.5', () => {
    expect(
      chooseSceneQuality({ viewportWidth: 1440, hardwareConcurrency: 12 }),
    ).toEqual({
      level: 'high',
      dpr: [1, 1.5],
    })
  })
})
