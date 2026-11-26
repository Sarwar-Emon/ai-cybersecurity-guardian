import { useEffect, useState } from 'react'

/** Shows a notice on the hosted demo. Renders nothing locally or if the check fails. */
export default function DemoBanner() {
  const [demo, setDemo] = useState(false)

  useEffect(() => {
    fetch('/api/v1/config')
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => setDemo(body?.demo === true))
      .catch(() => setDemo(false))
  }, [])

  if (!demo) return null

  return (
    <p
      role="note"
      className="border-b border-amber-300 bg-amber-100 px-6 py-2 text-center text-sm text-amber-900"
    >
      Public demo. Please do not paste real or private emails. It uses a shared daily limit, so it
      can stop for the day. Idle copies may take a minute to wake up.
    </p>
  )
}
