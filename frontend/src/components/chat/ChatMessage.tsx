import type { ReactNode } from 'react'

interface ChatMessageProps {
  role: 'user' | 'assistant'
  label: string
  children: ReactNode
}

export function ChatMessage({ role, label, children }: ChatMessageProps) {
  return (
    <article className={`message message--${role}`}>
      <div className="message-label">
        <span className="message-mark" aria-hidden="true">
          {role === 'assistant' ? '◇' : '■'}
        </span>
        {label}
      </div>
      <div className="message-body">{children}</div>
    </article>
  )
}
