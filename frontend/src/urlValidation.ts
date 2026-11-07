export function validateUrl(input: string): string | null {
  const value = input.trim()

  if (!value) {
    return 'Enter a URL to analyze.'
  }

  let parsedUrl: URL

  try {
    parsedUrl = new URL(value)
  } catch {
    return 'Enter a complete URL, such as https://example.com.'
  }

  if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
    return 'Only HTTP and HTTPS URLs are supported.'
  }

  const hostname = parsedUrl.hostname
  const isIPv6 = hostname.startsWith('[') && hostname.endsWith(']')

  if (
    !hostname ||
    (!hostname.includes('.') &&
      hostname !== 'localhost' &&
      !isIPv6)
  ) {
    return 'Enter a valid domain name or IP address.'
  }

  return null
}
