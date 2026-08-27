export interface HttpValidationError {
  ok: false;
  error: string;
  status: number;
}

export interface HttpValidationSuccess<T> {
  ok: true;
  value: T;
}

export type HttpValidationResult<T> = HttpValidationSuccess<T> | HttpValidationError;

function normalizedOrigin(value: string): string | null {
  try {
    const url = new URL(value);
    if (url.username || url.password || url.pathname !== '/' || url.search || url.hash) return null;
    return url.origin;
  } catch {
    return null;
  }
}

/**
 * Research mutations are deliberately same-origin only. SITE_ORIGIN is useful
 * when a production proxy makes request.url differ from the public site URL.
 */
export function validateResearchPostHeaders(
  request: Request,
  configuredSiteOrigin?: string,
): HttpValidationResult<undefined> {
  const mediaType = request.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase();
  if (mediaType !== 'application/json') {
    return { ok: false, error: 'Content-Type 必须是 application/json', status: 415 };
  }

  const requestOrigin = normalizedOrigin(new URL(request.url).origin);
  const expectedOrigin = configuredSiteOrigin
    ? normalizedOrigin(configuredSiteOrigin)
    : requestOrigin;
  if (!expectedOrigin) {
    return { ok: false, error: 'SITE_ORIGIN 配置无效', status: 503 };
  }

  const suppliedOrigin = request.headers.get('origin');
  if (!suppliedOrigin || normalizedOrigin(suppliedOrigin) !== expectedOrigin) {
    return { ok: false, error: '只接受同源研究请求', status: 403 };
  }

  const fetchSite = request.headers.get('sec-fetch-site');
  if (fetchSite && fetchSite !== 'same-origin') {
    return { ok: false, error: '只接受同源研究请求', status: 403 };
  }
  return { ok: true, value: undefined };
}

export async function readBoundedJson(
  request: Request,
  maxBytes: number,
): Promise<HttpValidationResult<unknown>> {
  const rawLength = request.headers.get('content-length');
  if (rawLength) {
    const parsedLength = Number(rawLength);
    if (Number.isFinite(parsedLength) && parsedLength > maxBytes) {
      return { ok: false, error: '请求过大', status: 413 };
    }
  }

  const body = request.body;
  if (!body) return { ok: false, error: 'JSON 格式无效', status: 400 };
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let raw = '';
  let receivedBytes = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      receivedBytes += chunk.value.byteLength;
      if (receivedBytes > maxBytes) {
        await reader.cancel('request body exceeds configured limit');
        return { ok: false, error: '请求过大', status: 413 };
      }
      raw += decoder.decode(chunk.value, { stream: true });
    }
    raw += decoder.decode();
  } catch {
    return { ok: false, error: '无法读取请求体', status: 400 };
  } finally {
    reader.releaseLock();
  }
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch {
    return { ok: false, error: 'JSON 格式无效', status: 400 };
  }
}
