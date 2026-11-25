// @vitest-environment jsdom

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import App from '../App'

afterEach(cleanup)

describe('Mode-specific dashboard information', () => {
  it('shows the correct detection methodology for URL mode', () => {
    render(<App />)

    expect(screen.getByText('Random Forest')).toBeTruthy()
    expect(screen.getByText('ML + Rules')).toBeTruthy()
    expect(screen.getByText('URL Threat Scanner')).toBeTruthy()
    expect(screen.getByText('URL Analyzer')).toBeTruthy()
    expect(screen.getByText('Machine learning and rule-based URL analysis.')).toBeTruthy()
    expect(screen.getByText('Analyze suspicious URLs with our machine learning engine.')).toBeTruthy()
    expect(screen.getByText('No scans yet')).toBeTruthy()
  })

  it('shows the correct detection methodology for Email mode', () => {
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Email Analysis' }))

    expect(screen.getByText('Heuristic Rules')).toBeTruthy()
    expect(screen.getByText('Content + Sender')).toBeTruthy()
    expect(screen.getByText('Email Threat Scanner')).toBeTruthy()
    expect(screen.getByText('Email Analyzer')).toBeTruthy()
    expect(screen.getByText('Heuristic email content and sender analysis.')).toBeTruthy()
    expect(screen.getByText('Analyze suspicious emails using heuristic detection rules.')).toBeTruthy()
    expect(screen.getByText('No scans yet')).toBeTruthy()

    expect(screen.queryByText('Random Forest')).toBeNull()
    expect(screen.queryByText('ML + Rules')).toBeNull()
    expect(screen.queryByText('URL Threat Scanner')).toBeNull()
  })
})
