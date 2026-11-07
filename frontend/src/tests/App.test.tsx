// @vitest-environment jsdom

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import App from '../App'

afterEach(() => {
  cleanup()
})

describe('Guardian AI dashboard', () => {
  it('renders the dashboard and URL scanner', () => {
    render(<App />)

    expect(
      screen.getByRole('heading', {
        name: 'Threat Detection Dashboard',
      })
    ).toBeTruthy()

    expect(
      screen.getByRole('textbox', {
        name: 'URL to analyze',
      })
    ).toBeTruthy()

    expect(
      screen.getByRole('button', {
        name: /Analyze URL/i,
      })
    ).toBeTruthy()
  })
})

import { fireEvent } from '@testing-library/react'
import { vi } from 'vitest'

describe('Guardian AI API integration', () => {
  it('displays a flagged ML result and security findings', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: 'analyzed',
        normalized: false,
        findings: [
          'URL uses HTTP instead of HTTPS',
          'Suspicious keyword in hostname: login',
        ],
        ml_analysis: {
          classification: 'flagged',
          phishing_class_score: 0.966,
          threshold: 0.5,
          model: 'Diverse Random Forest',
        },
        disclaimer: 'ML score is not a calibrated probability.',
      }),
    })

    vi.stubGlobal('fetch', mockFetch)

    try {
      render(<App />)

      fireEvent.change(screen.getByRole('textbox', {
        name: 'URL to analyze',
      }), {
        target: { value: 'http://example.com/login' },
      })

      fireEvent.click(screen.getByRole('button', {
        name: /Analyze URL/i,
      }))

      expect(
        await screen.findByText('Potential threat flagged')
      ).toBeTruthy()

      expect(screen.getByText('0.9660')).toBeTruthy()
      expect(screen.getByText('URL uses HTTP instead of HTTPS')).toBeTruthy()
      expect(screen.getByText('Suspicious keyword in hostname: login')).toBeTruthy()

      expect(mockFetch).toHaveBeenCalledOnce()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

describe('Guardian AI API error handling', () => {
  it('displays an error when the API returns HTTP 500', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'Internal server error' }),
    })

    vi.stubGlobal('fetch', mockFetch)

    try {
      render(<App />)

      fireEvent.change(
        screen.getByRole('textbox', { name: 'URL to analyze' }),
        { target: { value: 'https://example.com' } }
      )

      fireEvent.click(
        screen.getByRole('button', { name: /Analyze URL/i })
      )

      expect(
        await screen.findByText('Internal server error')
      ).toBeTruthy()

      expect(mockFetch).toHaveBeenCalledOnce()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

describe('Guardian AI network failure handling', () => {
  it('displays an error when the backend is unreachable', async () => {
    const mockFetch = vi.fn().mockRejectedValue(
      new Error('Failed to fetch')
    )

    vi.stubGlobal('fetch', mockFetch)

    try {
      render(<App />)

      fireEvent.change(
        screen.getByRole('textbox', { name: 'URL to analyze' }),
        { target: { value: 'https://example.com' } }
      )

      fireEvent.click(
        screen.getByRole('button', { name: /Analyze URL/i })
      )

      expect(
        await screen.findByText('Failed to fetch')
      ).toBeTruthy()

      expect(mockFetch).toHaveBeenCalledOnce()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

describe('Guardian AI input validation', () => {
  it('rejects invalid URLs without calling the backend', () => {
    const mockFetch = vi.fn()
    vi.stubGlobal('fetch', mockFetch)

    try {
      render(<App />)

      fireEvent.change(
        screen.getByRole('textbox', { name: 'URL to analyze' }),
        { target: { value: 'not-a-valid-url' } }
      )

      fireEvent.click(
        screen.getByRole('button', { name: /Analyze URL/i })
      )

      expect(mockFetch).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})

describe('Guardian AI not-flagged classification', () => {
  it('displays a not-flagged result without claiming the URL is safe', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: 'analyzed',
        normalized: false,
        findings: [],
        ml_analysis: {
          classification: 'not_flagged',
          phishing_class_score: 0.12,
          threshold: 0.5,
          model: 'Diverse Random Forest',
        },
        disclaimer: 'ML score is not a calibrated probability.',
      }),
    })

    vi.stubGlobal('fetch', mockFetch)

    try {
      render(<App />)

      fireEvent.change(
        screen.getByRole('textbox', { name: 'URL to analyze' }),
        { target: { value: 'https://example.com' } }
      )

      fireEvent.click(
        screen.getByRole('button', { name: /Analyze URL/i })
      )

      expect(
        await screen.findByText(/This does not guarantee safety/i)
      ).toBeTruthy()

      expect(screen.getByText('0.1200')).toBeTruthy()
      expect(mockFetch).toHaveBeenCalledOnce()
    } finally {
      vi.unstubAllGlobals()
    }
  })
})
