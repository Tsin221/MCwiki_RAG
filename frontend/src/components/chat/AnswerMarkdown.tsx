import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const ANSWER_ELEMENTS = [
  'p',
  'strong',
  'em',
  'ul',
  'ol',
  'li',
  'code',
  'pre',
  'blockquote',
  'br',
  'del',
  'table',
  'thead',
  'tbody',
  'tr',
  'th',
  'td',
] as const

interface AnswerMarkdownProps {
  text: string
  isStreaming: boolean
}

export function AnswerMarkdown({ text, isStreaming }: AnswerMarkdownProps) {
  return (
    <div className="answer-text" aria-busy={isStreaming}>
      <Markdown
        allowedElements={[...ANSWER_ELEMENTS]}
        remarkPlugins={[remarkGfm]}
        skipHtml
        unwrapDisallowed
      >
        {text || '正在准备回答…'}
      </Markdown>
      {isStreaming && <span className="stream-caret" aria-hidden="true" />}
    </div>
  )
}
