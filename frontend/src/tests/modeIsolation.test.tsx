// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Analysis mode isolation', () => {
  it('clears the previous status message when switching modes', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
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

    vi.stubGlobal('fetch', mockFetch)

    render(<App />)

    fireEvent.change(
      screen.getByRole('textbox', { name: 'URL to analyze' }),
      { target: { value: 'https://example.com/login' } }
    )

    fireEvent.click(screen.getByRole('button', { name: 'Analyze URL' }))

    await waitFor(() => {
      expect(screen.getByRole('status').textContent).toContain(
        'Potential threat detected'
      )
    })

    fireEvent.click(screen.getByRole('button', { name: 'Email Analysis' }))

    expect(screen.queryByText('ML Analysis')).toBeNull()
    expect(screen.queryByText('Security Findings')).toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
  })
})
