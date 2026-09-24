import type { SourceGroup } from '../../lib/sources'

export function SourceCard({ group }: { group: SourceGroup }) {
  const primary = group.segments[0]

  if (group.segments.length === 1) {
    return (
      <li className="source-card">
        <a href={primary.url} target="_blank" rel="noreferrer">
          <span className="source-index">[{primary.id}]</span>
          <span className="source-copy">
            <strong>{primary.title}</strong>
            <span>{primary.excerpt}</span>
          </span>
          <span className="source-arrow" aria-hidden="true">
            ↗
          </span>
        </a>
      </li>
    )
  }

  return (
    <li className="source-card source-card--group">
      <a className="source-link" href={group.url} target="_blank" rel="noreferrer">
        <span className="source-copy">
          <strong>{group.title}</strong>
        </span>
        <span className="source-segment-count">{group.segments.length} 段</span>
        <span className="source-arrow" aria-hidden="true">
          ↗
        </span>
      </a>
      <ul className="source-segments">
        {group.segments.map((segment) => (
          <li key={segment.id}>
            <span className="source-index">[{segment.id}]</span>
            <span className="source-copy">
              <span>{segment.excerpt}</span>
            </span>
          </li>
        ))}
      </ul>
    </li>
  )
}
