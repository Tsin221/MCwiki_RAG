import { describe, expect, it, vi } from 'vitest'

import type { AnswerStreamEvent } from '../types/api'
import { parseAnswerStream, requestAnswer } from './api'

function fragmentedStream(parts: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const part of parts) {
        controller.enqueue(encoder.encode(part))
      }
      controller.close()
    },
  })
}

describe('parseAnswerStream', () => {
  it('parses SSE events split across arbitrary byte chunks', async () => {
    const events: AnswerStreamEvent[] = []
    const stream = fragmentedStream([
      'event: meta\ndata: {"question":"红',
      '石是什么？"}\n\nevent: sources\r\ndata: {"items":[]}\r\n\r\n',
      'event: delta\ndata: {"text":"红石"}\n\n',
      'event: done\ndata: {"status":"answered"}\n\n',
    ])

    await parseAnswerStream(stream, (event) => events.push(event))

    expect(events).toEqual([
      { type: 'meta', question: '红石是什么？' },
      { type: 'sources', items: [] },
      { type: 'delta', text: '红石' },
      { type: 'done', status: 'answered' },
    ])
  })

  it('rejects known event names with invalid payloads', async () => {
    const stream = fragmentedStream([
      'event: delta\ndata: {"text":42}\n\n',
    ])

    await expect(parseAnswerStream(stream, () => undefined)).rejects.toThrow(
      'invalid delta event',
    )
  })
})

describe('requestAnswer', () => {
  it('posts JSON with credentials and consumes the response stream', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        fragmentedStream([
          'event: done\ndata: {"status":"answered"}\n\n',
        ]),
        {
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
        },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const events: AnswerStreamEvent[] = []
    await requestAnswer('红石是什么？', (event) => events.push(event))

    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/answers',
      expect.objectContaining({
        method: 'POST',
        credentials: 'include',
        body: JSON.stringify({ question: '红石是什么？' }),
      }),
    )
    expect(events).toEqual([{ type: 'done', status: 'answered' }])
    vi.unstubAllGlobals()
  })
})
