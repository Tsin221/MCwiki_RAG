import { createRoot } from 'react-dom/client'

import App from './App'
import './styles/tokens.css'
import './styles/app.css'

// React 19 StrictMode replays the R3F Canvas mount in development. With the
// current R3F renderer this can leave the reused WebGL context permanently lost.
createRoot(document.getElementById('root')!).render(<App />)
