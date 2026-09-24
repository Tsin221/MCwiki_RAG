import type { AnswerSource } from '../types/api'

export interface SourceGroup {
  url: string
  title: string
  segments: AnswerSource[]
}

/**
 * One card per page.
 *
 * A single article can contribute several evidence segments, each with its own
 * citation number, so the answer can cite [2] and [7] for the same page. Listing
 * the page once and keeping every segment under it leaves the citation numbers
 * intact while removing the repeated title from the list. Groups keep the order of
 * their first segment.
 */
export function groupSources(sources: AnswerSource[]): SourceGroup[] {
  const groups: SourceGroup[] = []
  const byUrl = new Map<string, SourceGroup>()
  for (const source of sources) {
    const existing = byUrl.get(source.url)
    if (existing) {
      existing.segments.push(source)
      continue
    }
    const group: SourceGroup = {
      url: source.url,
      title: source.title,
      segments: [source],
    }
    byUrl.set(source.url, group)
    groups.push(group)
  }
  return groups
}
