import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getReadiness, requestAnswer } from './lib/api'
import App from './App'

vi.mock('./lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./lib/api')>()
  return { ...actual, requestAnswer: vi.fn(), getReadiness: vi.fn() }
})

const requestAnswerMock = vi.mocked(requestAnswer)
const getReadinessMock = vi.mocked(getReadiness)

describe('App', () => {
  beforeEach(() => {
    requestAnswerMock.mockReset()
    getReadinessMock.mockReset()
    getReadinessMock.mockResolvedValue({ retrievalReady: true, answerReady: true })
  })

  it('shows the actual knowledge service state', async () => {
    getReadinessMock.mockResolvedValueOnce({ retrievalReady: false, answerReady: false })
    render(<App />)
    expect(await screen.findByText('知识库不可用')).toBeVisible()
    expect(screen.queryByText('知识库已连接')).not.toBeInTheDocument()
  })

  it('shows when retrieval works but the answer model is not configured', async () => {
    getReadinessMock.mockResolvedValueOnce({ retrievalReady: true, answerReady: false })
    render(<App />)
    expect(await screen.findByText('回答服务未配置')).toBeVisible()
  })

  it('submits a question and renders streamed text with its source', async () => {
    requestAnswerMock.mockImplementation(async (_question, onEvent) => {
      onEvent({ type: 'meta', question: '红石中继器有什么作用？' })
      onEvent({
        type: 'sources',
        items: [
          {
            id: 1,
            chunkId: 'redstone-repeater',
            title: '红石中继器',
            url: 'https://zh.minecraft.wiki/w/红石中继器',
            excerpt: '红石中继器可以延迟红石信号。',
          },
        ],
      })
      onEvent({ type: 'delta', text: '它可以延迟并增强' })
      onEvent({ type: 'delta', text: '红石信号。[1]' })
      onEvent({ type: 'done', status: 'answered' })
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(
      screen.getByLabelText('输入 Minecraft 问题'),
      '红石中继器有什么作用？',
    )
    await user.click(screen.getByRole('button', { name: '发送问题' }))

    expect(requestAnswerMock).toHaveBeenCalledWith(
      '红石中继器有什么作用？',
      expect.any(Function),
      expect.any(AbortSignal),
    )
    expect(await screen.findByText('它可以延迟并增强红石信号。[1]')).toBeVisible()
    expect(screen.getByRole('link', { name: /红石中继器/ })).toHaveAttribute(
      'href',
      'https://zh.minecraft.wiki/w/红石中继器',
    )
  })

  it('keeps the question and offers retry after a stream interruption', async () => {
    requestAnswerMock.mockRejectedValueOnce(new Error('connection lost'))
    requestAnswerMock.mockResolvedValueOnce()
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText('输入 Minecraft 问题'), '下界是什么？')
    await user.click(screen.getByRole('button', { name: '发送问题' }))

    const retry = await screen.findByRole('button', { name: '重新提问' })
    await user.click(retry)

    expect(requestAnswerMock).toHaveBeenLastCalledWith(
      '下界是什么？',
      expect.any(Function),
      expect.any(AbortSignal),
    )
  })

  it('renders streamed Markdown as semantic answer content', async () => {
    requestAnswerMock.mockImplementation(async (_question, onEvent) => {
      onEvent({ type: 'meta', question: 'Where do diamonds generate?' })
      onEvent({
        type: 'delta',
        text: 'Diamond ore generates:\n\n* **Below Y=16** [1]\n* More often near bedrock [2]',
      })
      onEvent({ type: 'done', status: 'answered' })
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByRole('textbox'), 'Where do diamonds generate?')
    await user.click(screen.getByRole('button', { name: '发送问题' }))

    expect(await screen.findByText('Below Y=16', { selector: 'strong' })).toBeVisible()
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    expect(screen.queryByText(/\*\*Below Y=16\*\*/)).not.toBeInTheDocument()
  })

  it('lists a page once when several of its segments are evidence', async () => {
    requestAnswerMock.mockImplementation(async (_question, onEvent) => {
      onEvent({ type: 'meta', question: '酿造台用什么燃料？' })
      onEvent({
        type: 'sources',
        items: [
          {
            id: 1,
            chunkId: 'brewing-stand-top',
            title: '酿造台',
            url: 'https://zh.minecraft.wiki/w/酿造台',
            excerpt: '酿造台使用烈焰粉作为燃料。',
          },
          {
            id: 2,
            chunkId: 'furnace',
            title: '烧炼',
            url: 'https://zh.minecraft.wiki/w/烧炼',
            excerpt: '一份燃料支持 20 次酿造。',
          },
          {
            id: 3,
            chunkId: 'brewing-stand-data',
            title: '酿造台',
            url: 'https://zh.minecraft.wiki/w/酿造台',
            excerpt: '酿造台的能量影响可执行配方的次数。',
          },
        ],
      })
      onEvent({ type: 'delta', text: '酿造台使用烈焰粉。[1]' })
      onEvent({ type: 'done', status: 'answered' })
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByRole('textbox'), '酿造台用什么燃料？')
    await user.click(screen.getByRole('button', { name: '发送问题' }))

    expect(await screen.findByText('3 条证据 · 2 个页面')).toBeVisible()
    const sources = screen.getByRole('region', { name: '参考来源' })
    expect(within(sources).getAllByRole('link')).toHaveLength(2)
    expect(within(sources).getByRole('link', { name: /酿造台/ })).toHaveAttribute(
      'href',
      'https://zh.minecraft.wiki/w/酿造台',
    )
    // Both segments of the page stay visible, each with its own citation number.
    expect(within(sources).getByText('2 段')).toBeVisible()
    expect(within(sources).getByText('酿造台使用烈焰粉作为燃料。')).toBeVisible()
    expect(
      within(sources).getByText('酿造台的能量影响可执行配方的次数。'),
    ).toBeVisible()
    expect(within(sources).getByText('[1]')).toBeVisible()
    expect(within(sources).getByText('[3]')).toBeVisible()
    expect(within(sources).getByText('[2]')).toBeVisible()
  })
})
