import type { FormEvent, KeyboardEvent } from 'react'

interface QuestionComposerProps {
  value: string
  isBusy: boolean
  onChange: (value: string) => void
  onSubmit: () => void
}

export function QuestionComposer({
  value,
  isBusy,
  onChange,
  onSubmit,
}: QuestionComposerProps) {
  const submit = (event: FormEvent) => {
    event.preventDefault()
    onSubmit()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (!isBusy && value.trim()) {
        onSubmit()
      }
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="question">
        输入 Minecraft 问题
      </label>
      <textarea
        id="question"
        value={value}
        rows={1}
        maxLength={1000}
        disabled={isBusy}
        placeholder="问一个关于 Minecraft 的问题…"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button
        className="send-button"
        type="submit"
        disabled={isBusy || !value.trim()}
        aria-label="发送问题"
      >
        <span aria-hidden="true">↑</span>
      </button>
      <p className="composer-hint">Enter 发送 · Shift + Enter 换行</p>
    </form>
  )
}
