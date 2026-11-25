// @vitest-environment jsdom

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import App from '../App'

afterEach(() => {
  cleanup()
})

describe('Guardian AI email analysis mode', () => {
  it('allows users to switch from URL to Email analysis', () => {
    render(<App />)

    expect(
      screen.getByRole('textbox', { name: 'URL to analyze' })
    ).toBeTruthy()

    fireEvent.click(
      screen.getByRole('button', { name: 'Email Analyzer' })
    )

    expect(
      screen.getByRole('textbox', { name: 'Email text to analyze' })
    ).toBeTruthy()

    expect(
      screen.getByRole('button', { name: 'Analyze Email' })
    ).toBeTruthy()
  })
})
