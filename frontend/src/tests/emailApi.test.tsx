// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import App from '../App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Guardian AI email API integration', () => {
  it('submits email text and displays heuristic risk findings', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'analyzed',
        score: 70,
        level: 'High Risk',
        findings: ['Account suspension or lockout threat detected.'],
        urls: [],
        sender_analysis: {
          findings: ['Possible sender domain impersonation'],
          score: 40,
        },
        content_score: 30,
        sender_score: 40,
        disclaimer: 'Email risk score is heuristic, not a calibrated phishing probability.',
      }),
    })

    vi.stubGlobal('fetch', mockFetch)

    render(<App />)

    fireEvent.click(
      screen.getByRole('button', { name: 'Email Analyzer' })
    )

    const emailText = [
      'From: PayPal Support <support@paypa1.com>',
      'Subject: Urgent verification',
      '',
      'Your account has been suspended.',
    ].join('\n')

    fireEvent.change(
      screen.getByRole('textbox', { name: 'Email text to analyze' }),
      { target: { value: emailText } }
    )

    fireEvent.click(
      screen.getByRole('button', { name: 'Analyze Email' })
    )

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/analyze/email',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ email_text: emailText }),
        })
      )
    })

    expect(await screen.findByText('High Risk')).toBeTruthy()
    expect(
      screen.getByText(/Account suspension or lockout threat detected\./)
    ).toBeTruthy()
  })
})
