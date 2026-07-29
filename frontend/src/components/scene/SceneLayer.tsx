import {
  Component,
  lazy,
  Suspense,
  type ErrorInfo,
  type ReactNode,
} from 'react'

import { useReducedMotion } from '../../hooks/useReducedMotion'
import type { ScenePhase } from '../../types/scene'
import SceneFallback from './SceneFallback'

const AmbientScene = lazy(() => import('./AmbientScene'))

interface BoundaryProps {
  children: ReactNode
}

interface BoundaryState {
  failed: boolean
}

class SceneErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { failed: false }

  static getDerivedStateFromError(): BoundaryState {
    return { failed: true }
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    // The static scene is the intentional recovery path. Avoid exposing GPU details.
  }

  render() {
    return this.state.failed ? <SceneFallback /> : this.props.children
  }
}

export function SceneLayer({ phase }: { phase: ScenePhase }) {
  const reducedMotion = useReducedMotion()
  if (reducedMotion) {
    return <SceneFallback />
  }

  return (
    <SceneErrorBoundary>
      <Suspense fallback={<SceneFallback />}>
        <AmbientScene phase={phase} />
      </Suspense>
    </SceneErrorBoundary>
  )
}
