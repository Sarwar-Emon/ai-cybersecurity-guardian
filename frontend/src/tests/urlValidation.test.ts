import { describe, expect, it } from 'vitest'
import { validateUrl } from '../urlValidation'

describe('Guardian AI URL validation', () => {
  it('rejects empty input', () => {
    expect(validateUrl('')).toBe('Enter a URL to analyze.')
  })

  it('rejects malformed URLs', () => {
    expect(validateUrl('not-a-valid-url')).not.toBeNull()
  })

  it('rejects unsupported protocols', () => {
    expect(validateUrl('ftp://example.com')).toBe(
      'Only HTTP and HTTPS URLs are supported.'
    )
  })

  it('accepts a valid HTTPS URL', () => {
    expect(validateUrl('https://example.com')).toBeNull()
  })

  it('accepts an IPv4 URL', () => {
    expect(validateUrl('http://127.0.0.1')).toBeNull()
  })

  it('accepts a bracketed IPv6 URL', () => {
    expect(validateUrl('http://[::1]')).toBeNull()
  })
})
