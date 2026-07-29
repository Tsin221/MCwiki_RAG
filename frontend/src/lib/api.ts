import type {
  AnswerSource,
  AnswerStatus,
  AnswerStreamEvent,
} from '../types/api'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ??
  'http://localhost:8000'

type EventHandler = (event: AnswerStreamEvent) => void

export class AnswerRequestError extends Error {
  readonly code: string
  readonly status: number

  constructor(message: string, code = 'NETWORK_ERROR', status = 0) {
    super(message)
    this.name = 'AnswerRequestError'
    this.code = code
    this.status = status
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isAnswerSource(value: unknown): value is AnswerSource {
  return (
    isRecord(value) &&
    typeof value.id === 'number' &&
    typeof value.chunkId === 'string' &&
    typeof value.title === 'string' &&
    typeof value.url === 'string' &&
    typeof value.excerpt === 'string'
  )
}

function isAnswerStatus(value: unknown): value is AnswerStatus {
  return value === 'answered' || value === 'insufficientEvidence'
}

function validatedEvent(
  type: string,
  payload: unknown,
): AnswerStreamEvent | null {
  if (!isRecord(payload)) {
    throw new AnswerRequestError(`invalid ${type} event`, 'INVALID_STREAM')
  }

  switch (type) {
    case 'meta':
      if (typeof payload.question !== 'string') {
        throw new AnswerRequestError('invalid meta event', 'INVALID_STREAM')
      }
      return { type, question: payload.question }
    case 'sources':
      if (
        !Array.isArray(payload.items) ||
        !payload.items.every(isAnswerSource)
      ) {
        throw new AnswerRequestError('invalid sources event', 'INVALID_STREAM')
      }
      return { type, items: payload.items }
    case 'delta':
      if (typeof payload.text !== 'string') {
        throw new AnswerRequestError('invalid delta event', 'INVALID_STREAM')
      }
      return { type, text: payload.text }
    case 'done':
      if (!isAnswerStatus(payload.status)) {
        throw new AnswerRequestError('invalid done event', 'INVALID_STREAM')
      }
      return { type, status: payload.status }
    case 'error':
      if (
        typeof payload.code !== 'string' ||
        typeof payload.message !== 'string'
      ) {
        throw new AnswerRequestError('invalid error event', 'INVALID_STREAM')
      }
      return { type, code: payload.code, message: payload.message }
    default:
      return null
  }
}

function parseEventBlock(block: string): AnswerStreamEvent | null {
  let type = ''
  const dataLines: string[] = []

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      type = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (!type || dataLines.length === 0) {
    return null
  }

  let payload: unknown
  try {
    payload = JSON.parse(dataLines.join('\n'))
  } catch {
    throw new AnswerRequestError(`invalid ${type} event`, 'INVALID_STREAM')
  }
  return validatedEvent(type, payload)
}

export async function parseAnswerStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: EventHandler,
): Promise<void> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let sawTerminalEvent = false

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })

    let boundary = buffer.match(/\r?\n\r?\n/)
    while (boundary?.index !== undefined) {
      const block = buffer.slice(0, boundary.index)
      buffer = buffer.slice(boundary.index + boundary[0].length)
      const event = parseEventBlock(block)
      if (event) {
        onEvent(event)
        sawTerminalEvent ||= event.type === 'done' || event.type === 'error'
      }
      boundary = buffer.match(/\r?\n\r?\n/)
    }

    if (done) {
      break
    }
  }

  if (buffer.trim()) {
    const event = parseEventBlock(buffer)
    if (event) {
      onEvent(event)
      sawTerminalEvent ||= event.type === 'done' || event.type === 'error'
    }
  }
  if (!sawTerminalEvent) {
    throw new AnswerRequestError(
      '回答连接意外中断，请重试。',
      'STREAM_INTERRUPTED',
    )
  }
}

async function errorFromResponse(response: Response): Promise<AnswerRequestError> {
  try {
    const payload: unknown = await response.json()
    if (
      isRecord(payload) &&
      isRecord(payload.error) &&
      typeof payload.error.code === 'string' &&
      typeof payload.error.message === 'string'
    ) {
      return new AnswerRequestError(
        payload.error.message,
        payload.error.code,
        response.status,
      )
    }
  } catch {
    // Fall through to the stable client-side message.
  }
  return new AnswerRequestError(
    '回答服务暂时不可用，请稍后重试。',
    'HTTP_ERROR',
    response.status,
  )
}

export async function requestAnswer(
  question: string,
  onEvent: EventHandler,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/answers`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new AnswerRequestError('无法连接回答服务，请检查网络后重试。')
  }

  if (!response.ok) {
    throw await errorFromResponse(response)
  }
  if (!response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new AnswerRequestError('回答服务返回了无效响应。', 'INVALID_STREAM')
  }
  if (!response.body) {
    throw new AnswerRequestError('回答服务未返回数据流。', 'INVALID_STREAM')
  }

  await parseAnswerStream(response.body, onEvent)
}
