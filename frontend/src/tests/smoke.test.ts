import { describe, expect, it } from 'vitest'

describe('Guardian AI test environment', () => {
  it('executes TypeScript tests correctly', () => {
    const supportedProtocols = ['http:', 'https:']

    expect(supportedProtocols).toContain('https:')
    expect(supportedProtocols).not.toContain('ftp:')
  })
})
