import { useEffect, useState } from 'react'

import { AnswerMarkdown } from './components/chat/AnswerMarkdown'
import { ChatMessage } from './components/chat/ChatMessage'
import { QuestionComposer } from './components/chat/QuestionComposer'
import { SourceCard } from './components/chat/SourceCard'
import { SceneLayer } from './components/scene/SceneLayer'
import { useAnswerStream } from './hooks/useAnswerStream'
import { getReadiness } from './lib/api'
import { groupSources } from './lib/sources'

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

const CONNECTION_COPY = {
  checking: '正在检查连接',
  ready: '知识库已连接',
  modelUnavailable: '回答服务未配置',
  unavailable: '知识库不可用',
} as const

export default function App() {
  const [input, setInput] = useState('')
  const [connection, setConnection] = useState<keyof typeof CONNECTION_COPY>('checking')
  const answer = useAnswerStream()

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const status = await getReadiness()
        if (active) {
          setConnection(
            !status.retrievalReady
              ? 'unavailable'
              : status.answerReady
                ? 'ready'
                : 'modelUnavailable',
          )
        }
      } catch {
        if (active) setConnection('unavailable')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 15_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const submit = () => {
    const question = input.trim()
    if (!question || answer.isBusy) {
      return
    }
    void answer.submit(question)
    setInput('')
  }

  const sourceGroups = groupSources(answer.sources)

  return (
    <div className="app-shell" data-phase={answer.phase}>
      <SceneLayer phase={answer.phase} />
      <a className="skip-link" href="#main-content">
        跳到主要内容
      </a>
      <header className="site-header">
        <a className="brand" href="/" aria-label="MC Wiki 助手首页">
          <span className="brand-cube" aria-hidden="true" />
          <span>
            <strong>MC.WIKI</strong>
            <small>探索者知识终端</small>
          </span>
        </a>
        <div className="knowledge-badge" data-status={connection} role="status">
          <span aria-hidden="true" />
          {CONNECTION_COPY[connection]}
        </div>
      </header>

      <main className="content" id="main-content">
        {!answer.question ? (
          <section className="welcome" aria-labelledby="welcome-title">
            <p className="eyebrow">
              <span>01</span>
              MINECRAFT KNOWLEDGE ARCHIVE
            </p>
            <h1 id="welcome-title">
              方块世界，
              <br />每一步都有
              <span>据可循。</span>
            </h1>
            <p className="welcome-copy">
              从中文 Minecraft Wiki 中检索、整理并回答你的问题。
              每条结论都附带出处，方便继续深入探索。
            </p>
            <dl className="knowledge-facts" aria-label="知识助手特性">
              <div>
                <dt>资料范围</dt>
                <dd>中文 Wiki</dd>
              </div>
              <div>
                <dt>回答方式</dt>
                <dd>实时生成</dd>
              </div>
              <div>
                <dt>信息依据</dt>
                <dd>来源可追溯</dd>
              </div>
            </dl>
            <div className="examples" aria-label="示例问题">
              {EXAMPLE_QUESTIONS.map((question) => (
                <button
                  type="button"
                  key={question}
                  onClick={() => setInput(question)}
                >
                  <span>{question}</span>
                  <i aria-hidden="true">↗</i>
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="conversation" aria-label="当前问答">
            <div className="conversation-heading" aria-hidden="true">
              <span>WIKI / RESPONSE</span>
              <i />
            </div>
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
                  <span>
                    {answer.sources.length} 条证据
                    {sourceGroups.length < answer.sources.length
                      ? ` · ${sourceGroups.length} 个页面`
                      : ''}
                  </span>
                </div>
                <ol>
                  {sourceGroups.map((group) => (
                    <SourceCard key={group.segments[0].id} group={group} />
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

      <div className="scene-note" aria-hidden="true">
        <span>OVERWORLD</span>
        <strong>暮色水域</strong>
        <i />
        <small>动态场景 · 01</small>
      </div>
    </div>
  )
}
