import { useState } from 'react'

import { AnswerMarkdown } from './components/chat/AnswerMarkdown'
import { ChatMessage } from './components/chat/ChatMessage'
import { QuestionComposer } from './components/chat/QuestionComposer'
import { SourceCard } from './components/chat/SourceCard'
import { SceneLayer } from './components/scene/SceneLayer'
import { useAnswerStream } from './hooks/useAnswerStream'

const EXAMPLE_QUESTIONS = [
  '红石中继器有什么作用？',
  '如何找到下界要塞？',
  '附魔台怎样达到最高等级？',
  '村民职业如何更换？',
]

const STATUS_COPY = {
  idle: '',
  retrieving: '正在检索 Wiki',
  generating: '正在组织回答',
  complete: '回答完成',
  insufficient: '现有资料不足',
  error: '回答失败',
  interrupted: '回答中断',
} as const

export default function App() {
  const [input, setInput] = useState('')
  const answer = useAnswerStream()

  const submit = () => {
    const question = input.trim()
    if (!question || answer.isBusy) {
      return
    }
    void answer.submit(question)
    setInput('')
  }

  return (
    <div className="app-shell" data-phase={answer.phase}>
      <SceneLayer phase={answer.phase} />
      <header className="site-header">
        <a className="brand" href="/" aria-label="MC Wiki 助手首页">
          <span className="brand-cube" aria-hidden="true">
            M
          </span>
          <span>
            <strong>MC WIKI</strong>
            <small>知识助手</small>
          </span>
        </a>
        <div className="knowledge-badge">
          <span aria-hidden="true" />
          本地知识库
        </div>
      </header>

      <main className="content">
        {!answer.question ? (
          <section className="welcome" aria-labelledby="welcome-title">
            <p className="eyebrow">MINECRAFT KNOWLEDGE ENGINE</p>
            <h1 id="welcome-title">
              从 Wiki 中找到
              <br />
              <span>有据可查</span>的答案
            </h1>
            <p className="welcome-copy">
              基于中文 Minecraft Wiki 的本地知识库。每个回答都附带可追踪来源，
              资料不足时会如实说明。
            </p>
            <div className="examples" aria-label="示例问题">
              {EXAMPLE_QUESTIONS.map((question) => (
                <button
                  type="button"
                  key={question}
                  onClick={() => setInput(question)}
                >
                  <span aria-hidden="true">＋</span>
                  {question}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="conversation" aria-label="当前问答">
            <ChatMessage role="user" label="你">
              <p>{answer.question}</p>
            </ChatMessage>
            <ChatMessage role="assistant" label="MC Wiki 助手">
              {answer.phase === 'retrieving' ? (
                <div className="retrieval-pulse">
                  <i />
                  <i />
                  <i />
                  <span>正在查找相关条目</span>
                </div>
              ) : (
                <AnswerMarkdown
                  text={answer.answer}
                  isStreaming={answer.phase === 'generating'}
                />
              )}
            </ChatMessage>

            {answer.sources.length > 0 && (
              <section className="sources" aria-labelledby="sources-title">
                <div className="section-heading">
                  <h2 id="sources-title">参考来源</h2>
                  <span>{answer.sources.length} 条证据</span>
                </div>
                <ol>
                  {answer.sources.map((source) => (
                    <SourceCard key={source.id} source={source} />
                  ))}
                </ol>
              </section>
            )}

            {(answer.phase === 'error' || answer.phase === 'interrupted') && (
              <div className="error-panel" role="alert">
                <div>
                  <strong>{STATUS_COPY[answer.phase]}</strong>
                  <p>{answer.errorMessage}</p>
                </div>
                <button
                  type="button"
                  onClick={() => void answer.submit(answer.question)}
                >
                  重新提问
                </button>
              </div>
            )}
          </section>
        )}
      </main>

      <div className="composer-dock">
        <div className="status-line" role="status" aria-live="polite">
          {STATUS_COPY[answer.phase] && (
            <>
              <span className="status-dot" aria-hidden="true" />
              {STATUS_COPY[answer.phase]}
            </>
          )}
        </div>
        <QuestionComposer
          value={input}
          isBusy={answer.isBusy}
          onChange={setInput}
          onSubmit={submit}
        />
        <p className="disclaimer">AI 回答可能有误，请以引用的 Wiki 原文为准。</p>
      </div>
    </div>
  )
}
