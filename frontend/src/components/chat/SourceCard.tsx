import type { AnswerSource } from '../../types/api'

export function SourceCard({ source }: { source: AnswerSource }) {
  return (
    <li className="source-card">
      <a href={source.url} target="_blank" rel="noreferrer">
        <span className="source-index">[{source.id}]</span>
        <span className="source-copy">
          <strong>{source.title}</strong>
          <span>{source.excerpt}</span>
        </span>
        <span className="source-arrow" aria-hidden="true">
          ↗
        </span>
      </a>
    </li>
  )
}
