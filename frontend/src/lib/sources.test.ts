import { describe, expect, it } from 'vitest'

import type { AnswerSource } from '../types/api'
import { groupSources } from './sources'

function source(id: number, url: string, title = '红石中继器'): AnswerSource {
  return {
    id,
    chunkId: `chunk-${id}`,
    title,
    url,
    excerpt: `第 ${id} 段的正文`,
  }
}

const REPEATER = 'https://zh.minecraft.wiki/w/红石中继器'
const COMPARATOR = 'https://zh.minecraft.wiki/w/红石比较器'

describe('groupSources', () => {
  it('leaves one segment per page as its own card', () => {
    const groups = groupSources([source(1, REPEATER), source(2, COMPARATOR)])

    expect(groups).toHaveLength(2)
    expect(groups[0].url).toBe(REPEATER)
    expect(groups[0].segments.map((segment) => segment.id)).toEqual([1])
  })

  it('keeps repeated segments of one page together in order', () => {
    const groups = groupSources([
      source(1, REPEATER),
      source(2, COMPARATOR, '红石比较器'),
      source(3, REPEATER),
      source(4, REPEATER),
    ])

    expect(groups).toHaveLength(2)
    expect(groups.map((group) => group.segments.map((seg) => seg.id))).toEqual([
      [1, 3, 4],
      [2],
    ])
    // The page keeps the position of its first segment and its first title.
    expect(groups[0].title).toBe('红石中继器')
  })

  it('returns nothing for no sources', () => {
    expect(groupSources([])).toEqual([])
  })
})
