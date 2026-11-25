// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Asynchronous request isolation', () => {
  it('ignores an outdated URL response after switching to Email mode', async () => {
    let resolveRequest!: (value: unknown) => void

    const pendingRequest = new Promise((resolve) => {
      resolveRequest = resolve
    })

    const mockFetch = vi.fn().mockReturnValue(pendingRequest)
    vi.stubGlobal('fetch', mockFetch)

    render(<App />)

    fireEvent.change(
      screen.getByRole('textbox', { name: 'URL to analyze' }),
      { target: { value: 'https://example.com/login' } }
    )

    fireEvent.click(screen.getByRole('button', { name: 'Analyze URL' }))

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Email Analysis' }))

    await act(async () => {
      resolveRequest({
        ok: true,
        status: 200,
        json: async () => ({
          status: 'analyzed',
          normalized: false,
          findings: ['Suspicious keyword in hostname'],
          ml_analysis: {
            classification: 'flagged',
            phishing_class_score: 0.96,
            threshold: 0.5,
            model: 'Diverse Random Forest',
          },
          disclaimer: 'Detection results are advisory.',
        }),
      })
      await pendingRequest
    })

    expect(
      screen.getByRole('textbox', { name: 'Email text to analyze' })
    ).toBeTruthy()

    expect(screen.queryByRole('status')).toBeNull()
    expect(screen.queryByText('Potential threat flagged')).toBeNull()
  })
})
