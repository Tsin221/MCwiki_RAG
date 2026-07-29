import { useCallback, useEffect, useRef, useState } from 'react'

import { AnswerRequestError, requestAnswer } from '../lib/api'
import type { AnswerSource, AnswerStreamEvent } from '../types/api'

export type AnswerPhase =
  | 'idle'
  | 'retrieving'
  | 'generating'
  | 'complete'
  | 'insufficient'
  | 'error'
  | 'interrupted'

interface AnswerState {
  phase: AnswerPhase
  question: string
  answer: string
  sources: AnswerSource[]
  errorMessage: string
}

const INITIAL_STATE: AnswerState = {
  phase: 'idle',
  question: '',
  answer: '',
  sources: [],
  errorMessage: '',
}

export function useAnswerStream() {
  const [state, setState] = useState<AnswerState>(INITIAL_STATE)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => () => controllerRef.current?.abort(), [])

  const submit = useCallback(async (rawQuestion: string) => {
    const question = rawQuestion.trim()
    if (!question) {
      return
    }

    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    let receivedText = ''
    setState({
      phase: 'retrieving',
      question,
      answer: '',
      sources: [],
      errorMessage: '',
    })

    const onEvent = (event: AnswerStreamEvent) => {
      switch (event.type) {
        case 'meta':
          setState((current) => ({ ...current, question: event.question }))
          break
        case 'sources':
          setState((current) => ({
            ...current,
            phase: 'generating',
            sources: event.items,
          }))
          break
        case 'delta':
          receivedText += event.text
          setState((current) => ({
            ...current,
            phase: 'generating',
            answer: current.answer + event.text,
          }))
          break
        case 'done':
          setState((current) => ({
            ...current,
            phase:
              event.status === 'answered' ? 'complete' : 'insufficient',
          }))
          break
        case 'error':
          setState((current) => ({
            ...current,
            phase: receivedText ? 'interrupted' : 'error',
            errorMessage: event.message,
          }))
          break
      }
    }

    try {
      await requestAnswer(question, onEvent, controller.signal)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      const message =
        error instanceof AnswerRequestError
          ? error.message
          : '回答连接意外中断，请稍后重试。'
      setState((current) => ({
        ...current,
        phase: receivedText ? 'interrupted' : 'error',
        errorMessage: message,
      }))
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
      }
    }
  }, [])

  return {
    ...state,
    isBusy: state.phase === 'retrieving' || state.phase === 'generating',
    submit,
  }
}
