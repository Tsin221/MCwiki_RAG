import { useEffect, useState } from 'react'

const QUERY = '(prefers-reduced-motion: reduce)'

function currentPreference() {
  return typeof window.matchMedia !== 'function' || window.matchMedia(QUERY).matches
}

export function useReducedMotion() {
  const [reducedMotion, setReducedMotion] = useState(currentPreference)

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') {
      return
    }
    const media = window.matchMedia(QUERY)
    const update = () => setReducedMotion(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  return reducedMotion
}
