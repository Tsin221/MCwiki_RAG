export interface AnswerSource {
  id: number
  chunkId: string
  title: string
  url: string
  excerpt: string
}

export type AnswerStatus = 'answered' | 'insufficientEvidence'

export type AnswerStreamEvent =
  | { type: 'meta'; question: string }
  | { type: 'sources'; items: AnswerSource[] }
  | { type: 'delta'; text: string }
  | { type: 'done'; status: AnswerStatus }
  | { type: 'error'; code: string; message: string }
